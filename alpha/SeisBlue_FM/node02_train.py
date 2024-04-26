# -*- coding: utf-8 -*-
from tqdm import tqdm
import glob
import mlflow.pytorch
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seisblue.io
from seisblue import generator, model

import torch
import mlflow
import mlflow.pytorch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall, Fbeta, Loss
from ignite.metrics import ConfusionMatrix
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import h5py

from seisblue import tool


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


def init_model(model, weight, device, learning_rate):
    model.to(device)
    class_weights = torch.tensor(weight).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5,
                                                  max_lr=1e-2)
    return model, loss_fn, optimizer, scheduler


def get_dataloader(train_dataset, val_dataset, batch_size):
    train_loader = generator.DualDataLoader(train_dataset,
                                            batch_size=batch_size,
                                            drop_last=True,
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=2,
                                             drop_last=True,
                                             shuffle=True)
    print(f'Get {len(train_loader)} batches for train.(contain flip data)')
    print(f'Get {len(val_loader)} batches for validation.')
    return train_loader, val_loader


def train_step(engine, batch, model, optimizer, loss_fn, scaler, device):
    model.train()
    optimizer.zero_grad()
    features, labels = batch
    features, labels = features.to(device), labels.to(device)
    with torch.cuda.amp.autocast():
        outputs = model(features)
        loss = loss_fn(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {'y_pred': outputs, 'y': labels, 'loss': loss.item()}


def valid_step(engine, batch, model, device):
    model.eval()
    with torch.no_grad():
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(features)
    return outputs, labels


def output_transform(output):
    y_pred, y = output
    y_pred_one_hot = torch.nn.functional.one_hot(torch.argmax(y_pred, dim=1),
                                                 num_classes=y_pred.shape[1])
    y_one_hot = torch.nn.functional.one_hot(torch.argmax(y, dim=1),
                                            num_classes=y.shape[1])
    return y_pred_one_hot, y_one_hot


def output_transform_argmax(output):
    y_pred, y = output
    return y_pred, y.argmax(dim=1)


def init_metrics(loss_fn):
    P = Precision(average=False, output_transform=output_transform_argmax)
    R = Recall(average=False, output_transform=output_transform_argmax)
    metrics = {
        'loss': Loss(loss_fn),
        'accuracy': Accuracy(output_transform=output_transform),
        'precision': P,
        'recall': R,
        'f1': Fbeta(beta=1.0, precision=P, recall=R, average=False)
    }
    return metrics


def run_training(train_loader, val_loader, model, device, num_epoch, weight,
                 learning_rate, experiment_name, run_name):
    mlflow.set_experiment(experiment_name)
    model, loss_fn, optimizer, scheduler = init_model(model, weight, device,
                                                      learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    trainer = Engine(
        lambda engine, batch: train_step(engine, batch, model, optimizer,
                                         loss_fn, scaler, device))
    evaluator = Engine(
        lambda engine, batch: valid_step(engine, batch, model, device))

    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x['loss']})

    class_names = ['U', 'D', 'X']
    cm_metric = ConfusionMatrix(num_classes=len(class_names),
                                output_transform=output_transform_argmax)
    cm_metric.attach(evaluator, 'cm')
    metrics = init_metrics(loss_fn)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_scheduler(trainer):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_evaluator_results(trainer):
        evaluator.run(val_loader)
        val_metrics = evaluator.state.metrics
        mertrics = {}
        for name, value in val_metrics.items():
            if name == 'cm':
                continue
            if torch.is_tensor(value):
                for i, v in enumerate(value):
                    mertrics[f"val_{name}_{class_names[i]}"] = float(v)
            else:
                mertrics[f"val_{name}"] = value

        mlflow.log_metrics(
            mertrics,
            step=evaluator.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=4))
    def log_confusion_matrix():
        cm = cm_metric.compute()
        fig = plot_confusion_matrix(cm, class_names=class_names)
        epoch_number = trainer.state.epoch
        filename = f'./figure/confusion_matrix_{epoch_number}.png'
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(filename, artifact_path="confusion_matrices")
        os.remove(filename)
        cm_metric.reset()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {"num_epochs": num_epoch, "learning_rate": learning_rate})
        trainer.run(train_loader, max_epochs=num_epoch)
        mlflow.pytorch.log_model(model, "model")


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='BuGn')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    config_data = seisblue.io.read_yaml('./config/data_config.yaml')
    config = seisblue.io.read_yaml('./config/model_config.yaml')
    c = config['train']
    run_name = c["run_name"]
    experiment_name = c["experiment_name"]
    fig_dir = f'./figure'
    tool.check_dir(fig_dir)
    batch_sizes, lr, epochs = c['batch_size'], float(c['lr']), c['epochs']
    weight = c["weight"]
    model = model.FMNet()

    train_dataset = generator.H5Dataset(
        glob.glob(f'./dataset/*train*.hdf5'),
        flip=True)
    val_dataset = generator.H5Dataset(
        glob.glob(f'./dataset/*valid*.hdf5'),
        flip=False)
    device = get_device()

    all_combination = list(itertools.product(batch_sizes, epochs))

    for (batch_size, num_epoch) in tqdm(all_combination):
        train_loader, val_loader = get_dataloader(train_dataset,
                                                  val_dataset,
                                                  batch_size)
        print("Start to train.")
        run_name = f'b{batch_size}_ep{num_epoch}_weight{weight}'
        run_training(train_loader, val_loader,
                     model, device, num_epoch,
                     weight, lr, experiment_name, run_name)
    mlflow.end_run()
