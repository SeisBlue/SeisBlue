import os
import argparse
import pandas as pd
from obspy import UTCDateTime

from seisnn.data.core import Instance
from seisnn.utils import get_config
from seisnn.data.io import read_dataset
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
    feature = Instance(example)
    picks = pd.DataFrame.from_dict({'pick_time': feature.pick_time,
                                    'pick_phase': feature.pick_phase,
                                    'pick_set': feature.pick_set})
    picks = picks.loc[picks['pick_set'] == "manual"]

    for i, p in picks.iterrows():
        pick_time = UTCDateTime(p['pick_time']) - UTCDateTime(feature.starttime)
        index = int(pick_time / feature.delta)

        trace = feature.trace[-1, :, 0]
        noise = trace[index - 100:index]
        signal = trace[index: index + 100]
        snr = signal_to_noise_ratio(signal, noise)
        pick_snr.append(snr)

    if n % 1000 == 0 and not n == 0:
        print(f'read {n} data')
    n += 1

plot_snr_distribution(pick_snr)
