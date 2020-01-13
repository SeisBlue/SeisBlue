import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

from seisnn.utils import get_config
from seisnn.io import read_dataset
from seisnn.core import Feature
from seisnn.example_proto import batch_iterator

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=False, help='dataset', type=str)
args = ap.parse_args()

config = get_config()
dataset_dir = os.path.join(config['DATASET_ROOT'], args.dataset)
dataset = read_dataset(dataset_dir)

for batch in dataset.shuffle(1000).batch(2):
    for example in batch_iterator(batch):
        feature = Feature(example)
        feature.plot(enlarge=True, snr=True)
