# -*- coding: utf-8 -*-
import argparse
import h5py
import numpy as np
from datetime import datetime
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import tempfile
import glob

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.autograd import Variable
import mlflow
import mlflow.pytorch
from itertools import chain
import scipy

from ignite.engine import Engine, Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from seisblue import SQL, plot, utils, io
import generator as gt
import core
from model import phasenet, transformer, GAN_transformer
import warnings
import matplotlib.cbook


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


def init_model(generator, discriminator, device, g_lr, d_lr):
    generator.to(device)
    discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr,
                                   betas=(0.5, 0.999))

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr,
                                   betas=(0.5, 0.999))
    adversarial_loss = torch.nn.BCELoss()
    # g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=5,
    #                                               gamma=0.5)
    # d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=5,
    #                                               gamma=0.5)
    return generator, discriminator, adversarial_loss, g_optimizer, d_optimizer


def init_metrics(loss_fn):
    metrics = {
        'loss': Loss(loss_fn)}
    return metrics


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


def get_dataset(filename):
    instances = []
    with h5py.File(filename, 'r') as f:
        for id, instance_h5 in f.items():
            instance = read_hdf5(instance_h5)
            max_label = max([label.data.max() for label in instance.labels])
            min_label = min([label.data.min() for label in instance.labels])

            if len(instance.labels) != 0 and max_label <= 1 and min_label >= 0 and len(
                    instance.features.data) == 3:
                instances.append(instance)
    return instances


def check_loader(loader, batch_size, model):
    for feature, label in loader:
        assert feature.shape == (batch_size, 3, 2048)
        assert label.shape == (batch_size, 3, 2048)
    try:
        output = model(feature)
        print(output.shape)
    except Exception as e:
        print(e)


def train_step(batch, generator, discriminator, adversarial_loss,
               g_optimizer, d_optimizer, device, step,
               epoch):
    features, labels = batch
    features, labels = features.to(device), labels.to(device)

    # calculate g_loss
    real = torch.full((features.size(0), 1), 1.0, device=device,
                      requires_grad=False)
    fake = torch.full((features.size(0), 1), 0.0, device=device,
                      requires_grad=False)
    g_optimizer.zero_grad()
    predicts = generator(features)
    concat_predicts = torch.cat((predicts, features), dim=2)
    g_loss = adversarial_loss(discriminator(concat_predicts), real)

    g_loss.backward()
    g_optimizer.step()

    # calculate d_loss
    d_optimizer.zero_grad()
    concatenated_real = torch.cat((labels, features), dim=2)
    concatenated_fake = torch.cat((predicts.detach(), features),
                                  dim=2)
    real_loss = adversarial_loss(discriminator(concatenated_real), real)
    real_loss.backward()
    fake_loss = adversarial_loss(discriminator(concatenated_fake), fake)
    fake_loss.backward()
    d_optimizer.step()
    d_loss = (fake_loss + real_loss) / 2

    return predicts, labels, features, g_loss.item(), d_loss.item()


def valid_step(batch, generator, adversarial_loss,
               device, step, epoch):
    generator.eval()
    with torch.no_grad():
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        outputs = generator(features)
        loss = adversarial_loss(outputs, labels)

    return outputs, labels, features, loss.item()


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


def run_training(train_loader, valid_loader,
                 generator, discriminator,
                 device, num_epoch, g_lr, d_lr,
                 experiment_name, run_name, w_gr, w_gdata,
                 batch_size):
    mlflow.set_experiment(experiment_name)
    generator.to(device)
    discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
    adversarial_loss = torch.nn.BCELoss()
    save_interval = 1

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {"num_epochs": num_epoch, "learning_rate_G": g_lr,
             "learning_rate_D": d_lr, "w_gr": w_gr, "w_gdata": w_gdata,
             "batch_size": batch_size})
        for epoch in tqdm(range(num_epoch)):
            train_g_loss, train_d_loss, valid_loss = 0.0, 0.0, 0.0
            generator.train()
            discriminator.train()
            for i, batch in enumerate(train_loader):
                features, labels = batch
                features, labels = features.to(device), labels.to(device)

                # calculate g_loss
                real = torch.full((features.size(0), 1), 1.0, device=device,
                                  requires_grad=False)
                fake = torch.full((features.size(0), 1), 0.0, device=device,
                                  requires_grad=False)
                g_optimizer.zero_grad()
                predicts = generator(features)
                concat_predicts = torch.cat((predicts, features), dim=2)
                gr_loss = adversarial_loss(discriminator(concat_predicts), real)
                gdata_loss = adversarial_loss(predicts, labels)

                g_loss = (w_gr * gr_loss + w_gdata * gdata_loss) / (
                            w_gr + w_gdata)
                g_loss.backward()
                g_optimizer.step()


                # calculate d_loss
                d_optimizer.zero_grad()
                concatenated_real = torch.cat((labels, features), dim=2)
                concatenated_fake = torch.cat((predicts.detach(), features),
                                              dim=2)
                real_loss = adversarial_loss(discriminator(concatenated_real),
                                             real)

                fake_loss = adversarial_loss(discriminator(concatenated_fake),
                                             fake)
                d_loss = (fake_loss + real_loss) / 2
                d_loss.backward()
                d_optimizer.step()

                train_g_loss += g_loss.item()
                train_d_loss += d_loss.item()

                if i % 10 == 0:
                    print(
                        f'Epoch {epoch}/{num_epoch}, iteration {i}/{len(train_loader)}, g_loss {g_loss}.')

                if i == 1:
                    y_pred = predicts.cpu().detach().numpy()
                    y = labels.cpu().detach().numpy()
                    x = features.cpu().detach().numpy()
                    plot_and_log_output(y_pred, y, x, epoch,
                                        dir_name='plots_train1',
                                        i=3)


            generator.eval()
            for j, batch in enumerate(valid_loader):
                predicts, labels, features, loss = valid_step(batch, generator,
                                                              adversarial_loss,
                                                              device, j, epoch)
                valid_loss += loss

                if j == 1:
                    y_pred = predicts.cpu().detach().numpy()
                    y = labels.cpu().detach().numpy()
                    x = features.cpu().detach().numpy()
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

            train_g_loss /= len(train_loader)
            mlflow.log_metric('train_g_loss', train_g_loss, step=epoch)

            train_d_loss /= len(train_loader)
            mlflow.log_metric('train_d_loss', train_d_loss, step=epoch)

            valid_loss /= len(valid_loader)
            mlflow.log_metric('valid_loss', valid_loss, step=epoch)


            if epoch % save_interval == 0:
                mlflow.pytorch.log_model(generator, f"model_{epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = io.read_yaml(args.data_config_filepath)
    c = config['train']
    database = c['database']
    run_name = c["run_name"]
    experiment_name = c["experiment_name"]
    batch_sizes, g_lr, d_lr, epochs, w_gr, w_gdata = c['batch_size'], float(
        c['g_lr']), float(c['d_lr']), c['epochs'], float(c['w_gr']), float(c['w_gdata'])

    generator = transformer.Transformer()
    discriminator = GAN_transformer.Discriminator()
    device = get_device()

    # filepaths = get_instance_filepaths(database, **c['instance_filter'])
    filepaths = list(sorted(glob.glob('/usr/src/app/dataset/TW*')))[:-30]
    print(f'Get {len(filepaths)} filepaths.')

    # instances = utils.parallel(filepaths,
    #                            func=get_dataset)
    # instances = list(chain.from_iterable(
    #     sublist for sublist in instances if sublist))
    # print(f'Get {len(instances)} instances.')
    #
    # train_ratio = 0.85
    # dataset_size = len(instances)
    # split = int(np.floor(train_ratio * dataset_size))
    # train_instances, val_instances = instances[:split], instances[split:]
    #
    # train_dataset = gt.PickDataset(train_instances)
    # valid_dataset = gt.PickDataset(val_instances)

    print("Start to train.")
    all_combination = list(itertools.product(batch_sizes, epochs))
    for (batch_size, num_epoch) in all_combination:
        # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
        #                           shuffle=True)
        # valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
        #                           shuffle=False)
        train_loader, valid_loader = get_loaders(gt.H5Dataset(filepaths),
                                                 batch_size=batch_size)
        run_training(train_loader, valid_loader,
                     generator, discriminator,
                     device, num_epoch, g_lr, d_lr,
                     experiment_name, run_name, w_gr, w_gdata,
                     batch_size)
