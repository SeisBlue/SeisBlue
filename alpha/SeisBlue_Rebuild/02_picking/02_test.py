# -*- coding: utf-8 -*-
import argparse
import h5py
import numpy as np
from datetime import datetime
from tqdm import tqdm
import itertools
from itertools import chain
import scipy
import obspy
import glob
import copy

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import mlflow
import mlflow.pytorch

import seisblue
import generator
import core
from model import phasenet, transformer


def get_instance_filepaths(database, **kwargs):
    client = seisblue.SQL.Client(database)
    waveforms = client.get_waveform(**kwargs)
    filepaths = list(set([waveform.datasetpath for waveform in waveforms]))
    print(f'Get {len(filepaths)} filepaths.')
    return filepaths


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

            if len(instance.labels) == 0:
                continue
            if max_label > 1 and min_label < 0:
                continue
            if len(instance.features.data) != 3:
                continue

            instance.labels[0].timewindow = instance.labels[
                                                0].timewindow or instance.timewindow
            instance.labels[0].picks = get_picks_by_threshold(
                instance.labels[0])
            if len(instance.labels[0].picks) < 2:
                continue
            if instance.labels[0].picks[0].phase != 'P':
                continue
            if instance.labels[0].picks[1].phase != 'S':
                continue
            # small_snr = False
            # for pick in instance.labels[0].picks:
            #     get_snr(instance, pick)
            #     if pick.snr < 10 and pick.tag == 'P':
            #         small_snr = True
            # if small_snr:
            #     continue

            instances.append(instance)
    return instances


def test(dataloader, model, device):
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for features in tqdm(dataloader):
            features = features.to(device)
            output = model(features)
            outputs.append(output.cpu())
    results = torch.cat(outputs, dim=0).numpy()
    return results


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


def save_pred_results(instances, pred_results, tag, phase, database, threshold,
                      distance=100, save=False):
    client = seisblue.SQL.Client(database)
    for i, instance in enumerate(instances):
        label = core.Label(tag=tag,
                           data=pred_results[i],
                           phase=phase,
                           timewindow=instance.timewindow)
        picks = get_picks_by_threshold(label, threshold, distance)
        for pick in picks:
            get_snr(instance, pick)
        label.picks = picks
        instance.labels.append(label)

        if save:
            with h5py.File(instance.datasetpath, "r+") as f:
                instance_h5 = f[instance.id]
                sub_grp = seisblue.io.enter_hdf5_sub_group(
                    instance_h5['labels'], tag)
                label = seisblue.utils.to_dict(label)
                seisblue.io.del_hdf5_layer(sub_grp)
                seisblue.io.write_hdf5_layer(sub_grp, label)
            client.add_picks(label.picks)
    return instances


def get_picks_by_threshold(label, threshold=0.5, distance=100,
                           from_ms=0, to_ms=-1):
    """
    Extract pick from label and write into the database.
    :param float height: Height threshold, from 0 to 1, default is 0.5.
    :param int distance: Distance threshold in data point.
    """
    picks = []
    for i, phase in enumerate(label.phase[0:2]):
        peaks, properties = scipy.signal.find_peaks(
            label.data[i, :], height=threshold,
            distance=distance
        )

        for j, peak in enumerate(peaks):
            if peak:
                pick_time = (
                        obspy.UTCDateTime(label.timewindow.starttime)
                        + (peak + from_ms) * label.timewindow.delta
                )

                picks.append(
                    core.Pick(
                        time=datetime.utcfromtimestamp(pick_time.timestamp),
                        inventory=label.inventory,
                        phase=label.phase[i],
                        tag=label.tag,
                        confidence=round(float(properties["peak_heights"][j]),
                                         2),
                    )
                )
    return picks


def precision_recall_f1_score(true_positive, pred_count, val_count):
    """
    Calculates precision, recall and f1 score.

    :param int true_positive: True positive count.
    :param int pred_count: Predict count.
    :param int val_count: Validation count.
    :rtype: float
    :return: (precision, recall, f1)
    """
    try:
        precision = true_positive / pred_count
        recall = true_positive / val_count
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError as e:
        print(e)
        precision, recall, f1 = 0, 0, 0

    return precision, recall, f1


def score(instances, delta=0.1, threshold=0.5,
          error_distribution=True, figdir=None):
    num_of_predict = {"P": 0, "S": 0}
    num_of_manual = {"P": 0, "S": 0}
    error_array = {"P": [], "S": []}
    true_positive = {"P": 0, "S": 0}
    metrics = {'P_precision': 0, 'P_recall': 0, 'S_precision': 0,
               'S_recall': 0}
    for instance in instances:
        manual_picks, predict_picks = [], []
        for i, label in enumerate(instance.labels):
            if len(label.picks) == 0:
                label.timewindow = label.timewindow or instance.timewindow
                label.picks = get_picks_by_threshold(label, threshold=threshold)
            if label.tag == 'manual':
                manual_picks = label.picks
            else:
                predict_picks = label.picks

        true_positive, error_array = judge_true_positive(
            manual_picks,
            predict_picks,
            delta,
            true_positive,
            error_array
        )
        for manual_pick in manual_picks:
            num_of_manual[manual_pick.phase] += 1
        for predict_pick in predict_picks:
            num_of_predict[predict_pick.phase] += 1

    seisblue.utils.check_dir(figdir, recreate=True)
    for phase in ["P", "S"]:
        precision, recall, f1 = precision_recall_f1_score(
            true_positive=true_positive[phase],
            val_count=num_of_manual[phase],
            pred_count=num_of_predict[phase],
        )
        if error_distribution:
            seisblue.plot.plot_error_distribution(error_array[phase], save_dir=fig_dir,
                                         phase=phase)
        print(f"num_{phase}_predict = {num_of_predict[phase]}")
        print(f"num_{phase}_label = {num_of_manual[phase]}")
        print(
            f"{phase}: precision = {precision},recall = {recall},f1 = {f1}")
        metrics[f'{phase}_precision'] = precision
        metrics[f'{phase}_recall'] = recall
    return metrics


def judge_true_positive(
        manual_picks,
        predict_picks,
        delta,
        true_positive,
        error_array,
):
    for manual_pick in manual_picks:
        for predict_pick in predict_picks:
            if predict_pick.phase == manual_pick.phase:
                time_diff = (
                        predict_pick.time - manual_pick.time).total_seconds()
                if -delta <= time_diff <= delta:
                    true_positive[predict_pick.phase] += 1
                error_array[predict_pick.phase].append(time_diff)

    return true_positive, error_array


def get_snr(instance, pick, second=1):
    try:
        vector = np.linalg.norm(instance.features.data, axis=0)
        point = int(
            (pick.time - instance.timewindow.starttime).total_seconds() * 100)
        if point >= second * 100:
            signal = vector[point: point + second * 100]
            noise = vector[point - len(signal): point]
        else:
            noise = vector[0:point]
            signal = vector[point: point + len(noise)]
        snr = signal_to_noise_ratio(signal=signal, noise=noise)
        pick.snr = np.around(snr, 4)
    except Exception as e:
        print(e)


def signal_to_noise_ratio(signal, noise):
    """
    Calculates power ratio from signal and noise.

    :param numpy.array signal: Signal trace data.
    :param numpy.array noise: Noise trace data.
    :rtype: float
    :return: Signal to noise ratseisblue.io.
    """
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(noise))
    snr = np.log10(signal_power / noise_power)
    return snr


if __name__ == '__main__':
    config = seisblue.utils.get_config("--data_config_filepath")
    c = config['test']

    database = c['database']
    batch_size = c['batch_size']
    # filepaths = get_instance_filepaths(database, **c['instance_filter'])
    filepaths = list(sorted(glob.glob(c['datasetpath'])))
    print(f'Get {len(filepaths)} filepaths.')
    raw_instances = seisblue.utils.parallel(filepaths,
                                   func=get_dataset)
    raw_instances = list(chain.from_iterable(
        sublist for sublist in raw_instances if sublist))
    print(f'Get {len(raw_instances)} instances.')

    dataset = generator.PickTestDataset(raw_instances)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False)
    device = get_device()

    fig_dir = f'/usr/src/app/02_picking/figure/{c["dataset"]}'
    instance_fig_dir = f'{fig_dir}/instances'
    seisblue.utils.check_dir(instance_fig_dir, recreate=True)

    epoch = c['model_dirname'].split('_')[-1]
    # epochs = range(5, 100, 5)

    # thresholds = np.arange(0.1, 1, 0.1)
    thresholds = [c['threshold']]

    deltas = [0.2]
    metrics_group = []
    for delta in deltas:
        metrics_all = {'P_precision': [], 'P_recall': [], 'S_precision': [],
                       'S_recall': []}
        for threshold in thresholds:
            instances = copy.deepcopy(raw_instances)
            model_uri = f"runs:/{c['RUN_ID']}/model_{epoch}"
            loaded_model = mlflow.pytorch.load_model(model_uri)

            print(f"Start to test. ({threshold=})")
            pred_results = test(dataloader, loaded_model, device)
            instances_updated = save_pred_results(instances.copy(), pred_results,
                                                  c['tag'],
                                                  c['phase'], database,
                                                  threshold=threshold)
            metrics = score(instances_updated, threshold=threshold, figdir=fig_dir,
                            delta=delta, error_distribution=True)
        #     for key, value in metrics.items():
        #         metrics_all[key].append(value)
        #     for instance in instances_updated[::100]:
        #         id = '.'.join([instance.timewindow.starttime.isoformat(),
        #                        instance.inventory.network,
        #                        instance.inventory.station])
        #         seisblue.plot.plot_dataset(instance, save_dir=fig_dir, title=f'{id}',
        #                           threshold=threshold)
        # metrics_group.append(metrics_all)
    # seisblue.plot.plot_metrics_by_epoch(metrics_all, epochs, fig_dir)
    # seisblue.plot.plot_metrics_by_threshold(metrics_group[0], thresholds,
    #                                         fig_dir,
    #                                         old_metrics=metrics_group[1],
    #                                         old_metrics_threshold=thresholds)

    # old_metrics = {
    #     'P_precision': [0.8478, 0.8657, 0.8796,
    #                     0.8965, 0.91156],
    #     'P_recall': [0.9524, 0.9479, 0.9425,
    #                  0.9346, 0.9245],
    #     'S_precision': [0.7914, 0.8033, 0.8104,
    #                     0.8177, 0.8279],
    #     'S_recall': [0.8679, 0.8631, 0.8594,
    #                  0.8548, 0.8460]},
    # old_metrics_threshold = [0.1, 0.2, 0.3, 0.4, 0.5]