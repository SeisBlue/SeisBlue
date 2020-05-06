import os

from seisnn.utils import get_config
from seisnn.io import read_dataset
from seisnn.core import Feature
from seisnn.example_proto import batch_iterator

dataset = 'test'

config = get_config()
dataset_dir = os.path.join(config['TFRECORD_ROOT'], dataset)
dataset = read_dataset(dataset_dir)

for batch in dataset.shuffle(1000).batch(2):
    for example in batch_iterator(batch):
        feature = Feature(example)
        feature.plot(enlarge=True, snr=True)
