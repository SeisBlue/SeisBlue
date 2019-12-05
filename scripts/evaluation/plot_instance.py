import tensorflow as tf

from seisnn.utils import  get_config
from seisnn.io import read_dataset
from seisnn.example_proto import extract_example
from seisnn.plot import plot_dataset

config = get_config()
dataset = read_dataset('test')

for example in dataset:
    example = tf.train.SequenceExample.FromString(example.numpy())
    feature = extract_example(example)

    plot_dataset(feature)

