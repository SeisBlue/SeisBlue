import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

from seisnn.core import Instance
from seisnn.utils import get_config
from seisnn.io import read_dataset
from seisnn.plot import plot_loss
from seisnn.example_proto import batch_iterator

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=False, help='model', type=str)
args = ap.parse_args()

config = get_config()
SAVE_MODEL_PATH = os.path.join(config['MODELS_ROOT'], args.model)
SAVE_HISTORY_PATH = os.path.join(SAVE_MODEL_PATH, 'history')
SAVE_PNG_PATH = os.path.join(SAVE_MODEL_PATH, 'png')

loss_log = os.path.join(SAVE_MODEL_PATH, f'{args.model}.log')
plot_loss(loss_log, SAVE_MODEL_PATH)

dataset = read_dataset(SAVE_HISTORY_PATH)
for batch in dataset.batch(2):
    for example in batch_iterator(batch):
        feature = Instance(example)
        feature.plot(title=feature.id, save_dir=SAVE_PNG_PATH)
        print(feature.id)
