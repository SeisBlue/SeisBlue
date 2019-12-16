import argparse

from seisnn.feature import Feature
from seisnn.utils import get_config
from seisnn.io import read_dataset

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='dataset', type=str)
args = ap.parse_args()

config = get_config()
dataset = read_dataset(args.dataset)

for example in dataset:
    feature = Feature(example)
    feature.filter_phase('P')
    feature.filter_channel('Z')
    feature.plot()
