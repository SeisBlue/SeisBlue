import os
import argparse

from seisnn.core import Feature
from seisnn.utils import get_config
from seisnn.io import read_dataset

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=False, help='model', type=str)
args = ap.parse_args()

config = get_config()
SAVE_LOG_PATH = os.path.join(config['MODELS_ROOT'], args.model, 'log')
SAVE_PNG_PATH = os.path.join(config['MODELS_ROOT'], args.model, 'png')
dataset_dir = os.path.join(SAVE_LOG_PATH, 'pre_train')
dataset = read_dataset(dataset_dir)

for example in dataset:
    feature = Feature(example)
    feature.plot(title=feature.id, save_dir=os.path.join(SAVE_PNG_PATH, 'pre_train'))
    print(feature.id)
