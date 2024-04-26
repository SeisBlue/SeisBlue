"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.2
"""
# from src.seisblue_pipeline.seisblue import core
from ...seisblue import core
import numpy as np
import h5py
import logging
from ...seisblue.model import Unet, PhaseNet
import torch
from torch import nn, cat, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy


def get_dataset_paths(params):
    database = params['database']
    db = core.Client(database)
    dataset_paths = db.get_dataset(to_date=params['to_date'],
                                   column=params['column'])
    dataset_paths = list(set([j for i in dataset_paths for j in i]))
    return dataset_paths


def read_hdf5(dataset_paths):
    log = logging.getLogger(__name__)
    instances = []
    dataset = []

    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as f:
            for instance_h5 in f.values():
                traces = []
                labels = []
                trace_id = instance_h5.attrs['trace_id']
                time_window = core.TimeWindow(
                    starttime=instance_h5['time_window'].attrs['starttime'],
                    endtime=instance_h5['time_window'].attrs['endtime'],
                    npts=instance_h5['time_window'].attrs['npts'],
                    delta=instance_h5['time_window'].attrs['delta'],
                    sampling_rate=instance_h5['time_window'].attrs[
                        'sampling_rate'],
                )
                inventory = core.Inventory(
                    network=instance_h5['inventory'].attrs['network'],
                    station=instance_h5['inventory'].attrs['station'],
                )
                dataset_features = torch.zeros(3,
                                               instance_h5['time_window'].attrs[
                                                   'npts'])
                for i, trace_h5 in enumerate(instance_h5['traces'].values()):
                    trace = core.Trace(
                        channel=trace_h5.attrs['channel'],
                        data=np.array(trace_h5['data']),
                    )
                    traces.append(trace)
                    dataset_features[i, :] = torch.tensor(
                        np.array(trace_h5['data']))

                for label_h5 in instance_h5['labels'].values():
                    picks = []
                    for pick_h5 in label_h5['picks'].values():
                        pick = core.Pick(
                            time=pick_h5.attrs['time'],
                            phase=pick_h5.attrs['phase'],
                        )
                        picks.append(pick)
                    label = core.Label(
                        picks=picks,
                        phase=list(label_h5.attrs['phase']),
                        tag=label_h5.attrs['tag'],
                        data=np.array(label_h5['data']),
                    )
                    labels.append(label)
                    dataset_labels = torch.tensor(np.array(label_h5['data']).T)

                instance = core.Instance(
                    inventory=inventory,
                    time_window=time_window,
                    traces=traces,
                    labels=labels,
                    trace_id=trace_id,
                )
                instances.append(instance)
                dataset.append((dataset_features, dataset_labels))
    log.debug(f"Get {len(instances)} instance for dataset.")
    return instances, dataset


def plot_dataset(dataset, **kwargs):
    instance = dataset[0]
    instance.plot(**kwargs)


def get_dataloader(params, dataset):
    batch_size = 64 if not params['batch_size'] else params['batch_size']
    test_split = 0.2 if not params['test_split'] else params['test_split']
    assert dataset[0][0].size() == dataset[0][1].size(), \
        f"feature and label don't have the same size.\n" \
        f"feature:{dataset[0][0].size()}, label{dataset[0][1].size()}"

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              shuffle=False,)
    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            shuffle=False,)

    return train_loader, val_loader


def loss_fn(y_pred, y_true, eps=1e-5):
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)
    h = h.mean()
    return -h


def train_loop(model, dataloader, optimizer):
    size = len(dataloader.dataset)
    for batch_id, (batch_X, batch_y) in enumerate(dataloader):
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * batch_X.shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(model, dataloader):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for (batch_X, batch_y) in dataloader:
            pred = model(batch_X)
            test_loss += loss_fn(pred, batch_y).item()

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")


def train_test(params, train_loader, val_loader):
    log = logging.getLogger(__name__)
    epochs = 5 if not params['epochs'] else params['epochs']
    lr = 1e-2 if not params['learning_rate'] else float(params['learning_rate'])

    model_name = params['model_name']
    log.debug(f'Use {model_name} model.')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # log.debug(f'Use {device} device.')

    model = PhaseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(model, train_loader, optimizer)
        test_loop(model, val_loader)

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, params['model_path'])
    log.debug(f"Save model in {params['model_path']}.")


def evaluate(params, dataloader):
    model_path = params['model_path']
    model = PhaseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    (batch_X, batch_y) = next(iter(dataloader))
    sample_X = batch_X[0]
    sample_y = batch_y[0]
    pred = model(sample_X.unsqueeze(0))
    pred_y = pred[0][:2, :].T.detach().numpy()

    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0,
                                                       "height_ratios": [3,
                                                                         1,
                                                                         1]})

    axs[0].plot(sample_X.T.numpy())
    axs[1].plot(sample_y.T.numpy()[:, :2])

    shape = 'triang'
    half_width = 20
    peaks_pred = np.zeros_like(pred_y)
    wavelet = scipy.signal.windows.get_window(shape, 2 * half_width)
    for i in range(2):
        peaks, _ = scipy.signal.find_peaks(pred_y[:, i], height=0.3, distance=100)
        peaks_pred[:, i][peaks] = pred_y[:, i][peaks]
        peaks_pred[:, i] = scipy.signal.convolve(peaks_pred[:, i], wavelet[1:], mode="same")
    axs[2].plot(peaks_pred)
    plt.show()
