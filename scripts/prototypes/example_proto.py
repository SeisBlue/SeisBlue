import os
import tensorflow as tf
from seisnn.io import read, write_tfrecord, read_dataset
from seisnn.utils import get_config
from seisnn.example_proto import trace_to_example, extract_trace_example

dataset = 'test'

st = read()
trace = st[0]

write_tfrecord(trace, dataset)

dataset = read_dataset(dataset)

for example in dataset:
    example = tf.train.Example.FromString(example.numpy())
    x = extract_trace_example(example)
    print(x)