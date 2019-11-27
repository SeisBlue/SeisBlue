import os
import tensorflow as tf
from seisnn.io import read, write_tfrecord
from seisnn.utils import get_config
from seisnn.example_proto import trace_to_example, extract_trace_example


st = read()
trace = st[0]

write_tfrecord(trace, 'test')

dataset = tf.data.Dataset('/home/jimmy/tfrecord/dataset/test')