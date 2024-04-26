# -*- coding: utf-8 -*-
import argparse
import h5py
import numpy as np
from datetime import datetime
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import glob

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import mlflow
import mlflow.pytorch
from ignite.engine import Engine, Events, create_supervised_trainer, \
    create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, Fbeta, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torchinfo import summary
from itertools import chain
import tempfile

from seisblue import SQL, plot, utils, io
import generator
import core
from model import phasenet, transformer


def get_instance_filepaths(database, **kwargs):
    client = SQL.Client(database)
    waveforms = client.get_waveform(**kwargs)
    filepaths = list(set([waveform.datasetpath for waveform in waveforms]))
    print(f'Get {len(filepaths)} filepaths.')
    return filepaths


def get_loaders(dataset, batch_size, train_ratio=0.85):
    dataset_size = len(dataset)
    print(f'Get {dataset_size} instances.')
    indices = list(range(dataset_size))
    split = int(np.floor(train_ratio * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True)
    valid_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f'Get {len(train_loader)} batches for train.')
    print(f'Get {len(valid_loader)} batches for validation.')
    return train_loader, valid_loader


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


def init_model(model, device, learning_rate):
    model.to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5,
    #                                               max_lr=1e-2)
    return model, loss_fn, optimizer


def init_metrics(loss_fn):
    metrics = {
        'loss': Loss(loss_fn, output_transform=lambda x: (x[0], x[1]))}
    # "accuracy": Accuracy(),
    return metrics


def plot_and_log_output(outputs, labels, features, epoch_num,
                        dir_name='plots', i=0, iteration=1):
    plt.figure(figsize=(6, 6))
    plt.suptitle(f"Epoch {epoch_num} Iter {iteration}")
    plt.subplot(4, 1, 2)
    plt.plot(features[i][0], color='black')
    plt.subplot(4, 1, 3)
    plt.plot(features[i][1], color='black')
    plt.subplot(4, 1, 4)
    plt.plot(features[i][2], color='black')
    plt.subplot(4, 1, 1)
    plt.plot(labels[i][0], label='manual P',
             color="#90CAF9")
    plt.plot(outputs[i][0], label='Transformer P',
             color="#2196F3")
    plt.plot(labels[i][1], label='manual S',
             color="#FFAB91")
    plt.plot(outputs[i][1], label='Transformer S',
             color="#FF5722")
    plt.legend()

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmpfile:
        tmpfile.name = f"plots_{epoch_num:0>6}_{iteration:0>10}.jpg"
        plt.savefig(tmpfile.name)
        mlflow.log_artifact(tmpfile.name, dir_name)
        plt.close()


def run_training(train_loader, val_loader, model, device, num_epoch,
                 learning_rate, experiment_name, run_name):
    mlflow.set_experiment(experiment_name)
    model, loss_fn, optimizer = init_model(model, device,
                                           learning_rate)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    def valid_step(engine, batch):
        model.eval()
        with torch.no_grad():
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

        return outputs, labels, features

    trainer = Engine(train_step)
    train_evaluator = Engine(valid_step)
    val_evaluator = Engine(valid_step)
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: x)

    metrics = init_metrics(loss_fn)
    for name, metric in metrics.items():
        metric.attach(train_evaluator, name)
    for name, metric in metrics.items():
        metric.attach(val_evaluator, name)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trainer_results(trainer):
        train_evaluator.run(train_loader)
        train_metrics = train_evaluator.state.metrics
        epoch_num = trainer.state.epoch
        metrics = {}

        for name, value in train_metrics.items():
            metrics[f"train_{name}"] = value
        mlflow.log_metrics(
            metrics,
            step=epoch_num)

        y_pred, y, x = train_evaluator.state.output
        plot_and_log_output(y_pred, y, x,
                            epoch_num,
                            dir_name='plots_train')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_evaluator_results(trainer):
        val_evaluator.run(val_loader)
        val_metrics = val_evaluator.state.metrics
        epoch_num = trainer.state.epoch
        metrics = {}

        for name, value in val_metrics.items():
            metrics[f"val_{name}"] = value
        mlflow.log_metrics(
            metrics,
            step=epoch_num)

        y_pred, y, x = val_evaluator.state.output
        plot_and_log_output(y_pred, y, x,
                            epoch_num,
                            dir_name='plots_valid')

    @trainer.on(Events.EPOCH_COMPLETED(every=200))
    def save_model(trainer):
        mlflow.pytorch.log_model(model,
                                 "model_epoch_{}".format(trainer.state.epoch))

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {"num_epochs": num_epoch, "learning_rate": learning_rate})
        trainer.run(train_loader, max_epochs=num_epoch)
        mlflow.pytorch.log_model(model, "model")


def read_hdf5(instance):
    labels = []
    timewindow = core.TimeWindow(
        starttime=datetime.fromisoformat(
            instance['timewindow'].attrs['starttime']),
        endtime=datetime.fromisoformat(instance['timewindow'].attrs['endtime']),
        npts=instance['timewindow'].attrs['npts'],
        delta=instance['timewindow'].attrs['delta']
    )
    inventory = core.Inventory(
        network=instance['inventory'].attrs['network'],
        station=instance['inventory'].attrs['station'],
    )

    stream = core.Stream(data=np.array(instance['features'].get('data')),
                         channel=instance['features'].attrs['channel'])

    for label_h5 in instance['labels'].values():
        label = core.Label(
            phase=label_h5.attrs['phase'],
            tag=label_h5.attrs['tag'],
            data=np.array(label_h5.get('data'))
        )
        labels.append(label)

    instance = core.Instance(
        inventory=inventory,
        timewindow=timewindow,
        features=stream,
        labels=labels,
    )

    return instance


def check_model(model, batch_size):
    x_batch = torch.randn(batch_size, 3, 2048)
    try:
        model.train()
        output = model(x_batch)
        print(output.shape)
    except Exception as e:
        print(e)


def check_loader(loader, batch_size, model):
    for feature, label in loader:
        assert feature.shape == (batch_size, 3, 2048)
        assert label.shape == (batch_size, 3, 2048)
    try:
        output = model(feature)
        print(output.shape)
    except Exception as e:
        print(e)


def get_dataset(filename):
    instances = []
    with h5py.File(filename, 'r') as f:
        for id, instance_h5 in f.items():
            instance = read_hdf5(instance_h5)
            max_label = max([label.data.max() for label in instance.labels])
            min_label = min([label.data.min() for label in instance.labels])

            if len(instance.labels) != 0 and max_label <= 1 and min_label >= 0 and len(instance.features.data) == 3:
                instances.append(instance)
    return instances


def is_fine_data(target_id):
    with open('/usr/src/app/02_picking/fine_id', 'r') as f:
        for line in f:
            current_id = line.strip()
            if current_id == target_id:
                return True
        return False


def check_data(filepath, fig_dir, num=10):
    with h5py.File(filepath, 'r') as f:
        for instance_h5 in list(f.values())[:num]:
            instance = read_hdf5(instance_h5)
            plot.plot_dataset(instance, save_dir=fig_dir)


def train_step(batch, model, optimizer, loss_fn, device):
    model.train()
    optimizer.zero_grad()
    features, labels = batch
    features, labels = features.to(device), labels.to(device)

    outputs = model(features)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    return outputs, labels, features, loss.item()


def valid_step(batch, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = loss_fn(outputs, labels)

    return outputs, labels, features, loss.item()


def run_train(train_loader, valid_loader, model, device, num_epoch,
              learning_rate, experiment_name, run_name):
    mlflow.set_experiment(experiment_name)
    model.to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {"num_epochs": num_epoch, "learning_rate": learning_rate})
        step = 1
        for epoch in range(num_epoch):
            train_loss, valid_loss = 0.0, 0.0
            model.train()
            for i, batch in enumerate(train_loader):
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_iter = loss.item()
                train_loss += train_loss_iter
                step += 1
                if i % 100 == 0:
                    print(f'Epoch {epoch}/{num_epoch}, iteration {i}/{len(train_loader)}, loss {train_loss_iter}.')
                if i == (len(train_loader)-1):
                    y_pred = outputs.cpu().detach().numpy()
                    y = labels.cpu().detach().numpy()
                    x = features.cpu().detach().numpy()
                    plot_and_log_output(y_pred, y, x, epoch,
                                        dir_name='plots_train1',
                                        i=11)
                    plot_and_log_output(y_pred, y, x, epoch,
                                        dir_name='plots_train2',
                                        i=16)
                    plot_and_log_output(y_pred, y, x, epoch,
                                        dir_name='plots_train3',
                                        i=19)
            train_loss /= len(train_loader)
            mlflow.log_metric('train_loss', train_loss, step=epoch)

            if epoch % 10 == 0:
                mlflow.pytorch.log_model(model,
                                         "model_epoch_{}".format(epoch))

            model.eval()
            with torch.no_grad():
                for j, batch_valid in enumerate(valid_loader):
                    features_valid, labels_valid = batch_valid
                    features_valid, labels_valid = features_valid.to(device), labels_valid.to(device)
                    outputs_valid = model(features_valid)
                    valid_loss_iter = loss_fn(outputs_valid, labels_valid)
                    valid_loss += valid_loss_iter
                    if j == (len(valid_loader) - 1):
                        y_pred = outputs_valid.cpu().detach().numpy()
                        y = labels_valid.cpu().detach().numpy()
                        x = features_valid.cpu().detach().numpy()
                        plot_and_log_output(y_pred, y, x, epoch,
                                            dir_name='plots_valid1',
                                            i=0)
                        plot_and_log_output(y_pred, y, x, epoch,
                                            dir_name='plots_valid2',
                                            i=3)
                        plot_and_log_output(y_pred, y, x, epoch,
                                            dir_name='plots_valid3',
                                            i=6)
                        plot_and_log_output(y_pred, y, x, epoch,
                                            dir_name='plots_valid4',
                                            i=9)
            valid_loss /= len(valid_loader)
            mlflow.log_metric('valid_loss', valid_loss, step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = io.read_yaml(args.data_config_filepath)
    c = config['train']
    database = c['database']
    run_name = c["run_name"]
    experiment_name = c["experiment_name"]
    batch_sizes, lr, epochs = c['batch_size'], float(c['g_lr']), c['epochs']

    model = transformer.Transformer()
    # summary(model, input_size=(batch_sizes[0], 3, 2048))

    device = get_device()

    # filepaths = get_instance_filepaths(database, **c['instance_filter'])
    filepaths = list(sorted(glob.glob('/usr/src/app/dataset/*')))
    print(f'Get {len(filepaths)} filepaths.')

    # # plot waveform
    # fig_dir = './02_picking/figure/waveform/'
    # utils.check_dir(fig_dir, recreate=True)
    # for filepath in tqdm(filepaths):
    #     check_data(filepath, fig_dir=fig_dir, num=10)


    print("Start to train.")
    all_combination = list(itertools.product(batch_sizes, epochs))
    for (batch_size, num_epoch) in tqdm(all_combination):
        train_loader, valid_loader = get_loaders(generator.H5Dataset(filepaths),
                                                 batch_size=batch_size)
        run_train(train_loader, valid_loader, model, device, num_epoch,
                  lr, experiment_name, run_name)
