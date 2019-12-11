import tensorflow as tf

from seisnn.feature import select_phase, select_channel
from seisnn.utils import  get_config
from seisnn.io import read_dataset
from seisnn.example_proto import extract_example
from seisnn.plot import plot_dataset

config = get_config()
dataset = read_dataset('test')

for example in dataset:
    example = tf.train.SequenceExample.FromString(example.numpy())
    feature = extract_example(example)
    feature = select_phase(feature, 'P')
    feature = select_channel(feature, 'Z')
    plot_dataset(feature)

