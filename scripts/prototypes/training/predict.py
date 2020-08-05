import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

import tensorflow as tf

from seisnn.utils import get_config, make_dirs
from seisnn.data.io import read_dataset, write_tfrecord
from seisnn.data.core import parallel_to_tfrecord
from seisnn.data.example_proto import batch_iterator
from seisnn.model.settings import model, optimizer

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='input dataset', type=str)
ap.add_argument('-o', '--output', required=True, help='output dataset', type=str)
ap.add_argument('-m', '--model', required=True, help='model', type=str)
args = ap.parse_args()

config = get_config()

MODEL_PATH = os.path.join(config['MODELS_ROOT'], args.model)
make_dirs(MODEL_PATH)

OUTPUT_DATASET = os.path.join(config['DATASET_ROOT'], args.output)
make_dirs(OUTPUT_DATASET)

INPUT_DATASET = os.path.join(config['DATASET_ROOT'], args.input)
dataset = read_dataset(INPUT_DATASET)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, MODEL_PATH, max_to_keep=100)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = len(ckpt_manager.checkpoints)
    print(f'Latest checkpoint epoch {last_epoch} restored!!')

n = 0
for batch in dataset.take(1000).batch(512).prefetch(2):
    pdf = model.predict(batch['trace'])
    batch['pdf'] = tf.concat([batch['pdf'], pdf], axis=3)

    phase = batch['phase'].to_list()
    for p in phase:
        if not [b'p'] in p:
            p.append([b'p'])
    batch['phase'] = tf.ragged.constant(phase)

    with tf.device('/cpu:0'):
        example_list = parallel_to_tfrecord(list(batch_iterator(batch)))
        filename = f'{args.output}_{n}.tfrecord'
        write_tfrecord(example_list, os.path.join(OUTPUT_DATASET, filename))
    print(f'save {filename}')
    n += 1


