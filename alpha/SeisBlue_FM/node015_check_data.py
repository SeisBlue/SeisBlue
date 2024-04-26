# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import glob
import h5py
from tqdm import tqdm
import numpy as np
import random

from seisblue import tool, io, generator


def data_inspector(event_instance, figdir='./figure/data_inspector'):
    sampled_items = random.sample(event_instance.instances, k=2)
    for instance in sampled_items:
        features = np.array([trace.data for trace in instance.traces if
                    list(trace.channel)[-1] == 'Z'])
        manual, pred, confidence, raw_pred_label = None, None, None, None
        for label in instance.labels:
            if label.tag == 'manual':
                manual = label.pick.polarity
            if label.tag == 'SeisPolar':
                pred = label.pick.polarity
                raw_pred_label = label.pick.raw_polarity
                confidence = label.pick.confidence

        plt.figure()
        plt.plot(np.squeeze(features), c='k', label=instance.id)
        plt.axvline(156, c='red', label='P arrival', lw=1)
        plt.legend(loc='upper left')
        note = f'(raw: {raw_pred_label})' if raw_pred_label and raw_pred_label != pred else ''

        plt.title(f'{manual=}, {pred=}{note} ({confidence=:.2f})')
        plt.savefig(f'{figdir}/{instance.id}.jpg')
        plt.close()


if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['global']
    filepath = glob.glob(f'./dataset/{c["dataset_name"]}*test*.hdf5')[0]
    figdir = f'./figure/data_inspector/{c["dataset_name"]}'
    tool.check_dir(figdir, recreate=True)
    with h5py.File(filepath, "r") as f:
        sampled_items = random.sample(list(f.values()), k=30)
        for event_instance_h5 in tqdm(sampled_items):
            event_instance = io.read_hdf5_instance(event_instance_h5)
            data_inspector(event_instance, figdir=figdir)
