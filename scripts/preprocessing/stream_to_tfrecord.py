from multiprocessing import cpu_count
import tensorflow as tf

from seisnn.io import database_to_tfrecord

print(f'cpu counts: {cpu_count()} threads')

database = 'test.db'

with tf.device('/cpu:0'):
    database_to_tfrecord(database, 'test')
