import os
import argparse

from seisnn.core import Feature
from seisnn.utils import get_config
from seisnn.io import read_dataset
from seisnn.qc import signal_to_noise_ratio
from seisnn.plot import plot_snr_distribution

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=False, help='dataset', type=str)
args = ap.parse_args()

config = get_config()
dataset_dir = os.path.join(config['DATASET_ROOT'], args.dataset)
dataset = read_dataset(dataset_dir).shuffle(100000).prefetch(10)

pick_snr = []
n = 0
for example in dataset:
    feature = Feature(example)
    feature.filter_phase('P')
    picks = feature.picks.loc[feature.picks['pick_set'] == 'manual']

    for i, p in picks.iterrows():
        pick_time = p['pick_time'] - feature.starttime
        index = int(pick_time / feature.delta)
        for k, v in feature.channel.items():
            try:
                noise = v[index - 100:index]
                signal = v[index: index + 100]
                snr = signal_to_noise_ratio(signal, noise)
                pick_snr.append(snr)
            except IndexError:
                pass
    if n % 1000 == 0 and not n == 0:
        print(f'read {n} data')
    n += 1

plot_snr_distribution(pick_snr)
