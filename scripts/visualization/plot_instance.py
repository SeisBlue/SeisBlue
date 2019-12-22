import os
import argparse

from seisnn.core import Feature
from seisnn.utils import get_config
from seisnn.io import read_dataset

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=False, help='dataset', type=str)
args = ap.parse_args()

config = get_config()
dataset_dir = os.path.join(config['DATASET_ROOT'], args.dataset)
dataset = read_dataset(dataset_dir).shuffle(100000).prefetch(10)

for example in dataset:
    feature = Feature(example)
    feature.filter_phase('P')
    feature.filter_channel('Z')
    feature.plot()
