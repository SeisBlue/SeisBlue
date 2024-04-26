# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mlflow
import os
from collections import Counter
from matplotlib.ticker import MultipleLocator, FixedLocator
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

import seisblue.model
from seisblue import generator, core, tool, io


def get_dataloader(dataset, batch_size, shuffle=False, mode=None):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         drop_last=True,
                                         shuffle=shuffle)

    print(f'Get {len(loader)} batches of {mode} instances.')
    return loader


def test(dataloader, model, model_path, device):
    model.to(device)
    y_pred = np.array([])
    y_confidence = np.array([])

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        for (features, label) in tqdm(dataloader):
            features = features.to(device)
            outputs = model(features)
            test_confidence, test_pred = torch.max(outputs, 1)
            y_pred = np.concatenate((y_pred, test_pred.cpu().numpy()), axis=0)
            y_confidence = np.concatenate(
                (y_confidence, test_confidence.cpu().numpy()), axis=0)
    return y_pred, y_confidence


def test_mlflow(dataloader, model, device):
    model.to(device)
    y_pred = np.array([], dtype=np.int32)
    y_confidence = np.array([])
    batch_outputs = []

    model.eval()
    with torch.no_grad():
        for (features, label) in tqdm(dataloader):
            features = features.to(device)
            outputs = model(features)
            batch_outputs.append(outputs)
            test_confidence, test_pred = torch.max(outputs, 1)
            y_pred = np.concatenate((y_pred, test_pred.cpu().numpy()), axis=0)
            y_confidence = np.concatenate(
                (y_confidence, test_confidence.cpu().numpy()), axis=0)
        print(outputs)
        y_outputs = torch.cat(batch_outputs, dim=0)
        rounded_y_ouputs = torch.round(y_outputs * 100) / 100

    return y_pred, y_confidence, rounded_y_ouputs.cpu().numpy()


def get_predict_label(pred, confidence, y_outputs, filepaths, tag, threshold=0):
    polarity_columns = ['positive', 'negative', 'undecidable']
    pred = pred.tolist()
    confidence = confidence.tolist()
    num_y_pred = []
    num_y_true = []
    y_pred = []
    y_true = []
    events_id = []
    for archive in filepaths:
        with h5py.File(archive, "r+") as f:
            for event in f.values():
                num_y_pred_one_event = 0
                num_y_true_one_event = 0
                for i, instance in enumerate(event['instances'].values()):
                    if len(pred) == 0:
                        break
                    pred_y = pred.pop(0)
                    value = confidence.pop(0)
                    score = y_outputs[i]
                    pred_polarity_raw = polarity_columns[pred_y]
                    if value < threshold:
                        pred_polarity = 'undecidable'
                    else:
                        pred_polarity = pred_polarity_raw

                    if pred_polarity != 'undecidable':
                        num_y_pred_one_event += 1

                    sub_grp = io.enter_hdf5_sub_group(
                        instance['labels'], tag)
                    pick = core.Pick(
                        **dict(instance['labels/manual/pick'].attrs))
                    timewindow = core.TimeWindow(**dict(instance['timewindow']).attrs)
                    data = np.array([trace_h5['data'] for trace_h5 in instance['traces'].values()])
                    if pick.polarity != 'undecidable':
                        num_y_true_one_event += 1
                    y_true.append(pick.polarity)

                    pick.polarity = pred_polarity
                    pick.tag = tag
                    pick.confidence = value
                    pick.raw_polarity = pred_polarity_raw
                    get_snr(pick, data, timewindow)
                    label = core.Label(pick=pick, tag=tag, data=score)
                    label = tool.to_dict(label)
                    io.del_hdf5_layer(sub_grp)
                    io.write_hdf5_layer(sub_grp, label)
                    y_pred.append(pred_polarity)


                num_y_pred.append(num_y_pred_one_event)
                num_y_true.append(num_y_true_one_event)
                events_id.append(dict(event.attrs)['id'])
    print('Add predict labels.')
    return num_y_pred, num_y_true, events_id, y_pred, y_true


def get_snr(pick, data, timewindow, second=1):
    try:
        vector = np.linalg.norm(data, axis=0)
        point = int((pick.time - timewindow.starttime).total_seconds() * 100)
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
    :return: Signal to noise ratio.
    """
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(noise))
    snr = np.log10(signal_power / noise_power)
    return snr


def plot_histogram(num_y_pred, num_y_true, events_id,
                   title='Polarity Numbers of Event',
                   fig_dir='.'):
    print("Plot polarity numbers of event.")
    fig, ax = plt.subplots(figsize=(25, 6))
    bar_width = 0.3
    offset = bar_width / 2
    ax.bar(np.arange(len(num_y_true)) - offset, num_y_true, color='lightblue',
           width=bar_width,
           label='manual')
    ax.bar(np.arange(len(num_y_pred)) + offset, num_y_pred,
           color='crimson', width=bar_width, label='predict')

    ax.set_xlabel("Event")
    ax.set_ylabel("Count")
    ax.set_xticks(ticks=np.arange(len(num_y_pred)), labels=events_id,
                  rotation=65)
    ax.set_xticklabels(events_id, rotation=65, ha='right', va='top')
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'polarity_numbers_of_event.png'))


def plot_score_histogram(label, score, threshold, title='', fig_dir='.'):
    u, d, x = [], [], []
    for (l, s) in zip(label, score):
        if l == 0:
            u.append(s)
        elif l == 1:
            d.append(s)
        elif l == 2:
            x.append(s)
    print("Plot distribution of predict score.")
    plt.figure(figsize=(8, 8))
    plt.suptitle(f'{title}\n')
    plt.subplot(3, 1, 1)
    plt.hist(u, 20, label='Test set')
    plt.ylabel("Count")
    plt.title('Positive')
    plt.axvline(threshold, color='k', linestyle='--', label='threshold')
    min_x = min(score)
    plt.axvspan(min_x, threshold, color='white', alpha=0.5)
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2, sharex=plt.gca())
    plt.hist(d, 20)
    plt.ylabel("Count")
    plt.title('Negative')
    plt.axvline(threshold, color='k', linestyle='--', label='threshold')
    plt.axvspan(min_x, threshold, color='white', alpha=0.5)

    plt.subplot(3, 1, 3, sharex=plt.gca())
    plt.hist(x, 20)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title('Undecidable')
    plt.axvspan(min_x, 1, color='white', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'distribution_of_predict_score.png'),
                dpi=300)


def plot_confusion_matrix(y_pred, y_true, title='confusion_matrix',
                          fig_dir='.'):
    cf_matrix = confusion_matrix(y_true, y_pred)
    per_cls_acc = cf_matrix.diagonal() / cf_matrix.sum(axis=0)
    class_names = ['Positive', 'Negative', ' Undecidable']
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)
    print(f'Per calss accuracy: {per_cls_acc} {class_names}')

    plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.title(title)
    plt.savefig(os.path.join(fig_dir, f'{title}.png'))


def main():
    config_data = io.read_yaml('./config/data_config.yaml')
    cd = config_data['process_event']
    config = io.read_yaml('./config/model_config.yaml')
    c = config['test']
    filepahts = glob.glob(f'./dataset/{cd["dataset_name"]}*test*.hdf5')
    test_dataset = generator.H5Dataset(filepahts)
    print(f'Get {len(test_dataset)} of test instances.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use {device} device.')

    model = seisblue.model.FMNet()
    model_uri = f"runs:/{c['RUN_ID']}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri)

    print("Start to test.")
    test_loader = get_dataloader(test_dataset, c['batch_size'], mode='test')
    pred, confidence, y_outputs = test_mlflow(test_loader, loaded_model, device)
    # pred, confidence = test(test_loader, model, c['model_filepath'], device)
    # print(Counter(pred))

    num_y_pred, num_y_true, events_id, y_pred, y_true = get_predict_label(pred,
                                                                          confidence,
                                                                          y_outputs,
                                                                          filepahts,
                                                                          c['tag'],
                                                                          c['threshold'])

    fig_dir = f'./figure/{cd["dataset_name"]}'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plot_histogram(num_y_pred, num_y_true, events_id,
                   title=f'Polarity Numbers of Event for {cd["dataset_name"]}',
                   fig_dir=f'./figure/{cd["dataset_name"]}')

    plot_score_histogram(pred,
                         confidence,
                         c['threshold'],
                         title=f'Polarity Numbers of predict score for {cd["dataset_name"]}',
                         fig_dir=f'./figure/{cd["dataset_name"]}')

    plot_confusion_matrix(y_pred, y_true,
                          title=f'Confusion Matrix {cd["dataset_name"]}',
                          fig_dir=f'./figure/{cd["dataset_name"]}')

if __name__ == '__main__':
    main()
