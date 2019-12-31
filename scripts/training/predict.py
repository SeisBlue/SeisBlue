import os
import argparse

import tensorflow as tf

from seisnn.utils import get_config, make_dirs
from seisnn.io import read_dataset
from seisnn.core import Feature

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
dataset = read_dataset(INPUT_DATASET).take(1000)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, MODEL_PATH, max_to_keep=100)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = len(ckpt_manager.checkpoints)
    print(f'Latest checkpoint epoch {last_epoch} restored!!')

for example in dataset:
    feature = Feature(example)
    feature.filter_phase('P')
    feature.filter_channel('Z')

    trace = feature.get_trace()
    feature.phase['predict'] = model.predict(trace)[0, 0, -3001:, 0]
    feature.get_picks('predict')
    filename = f'{feature.starttime}_{feature.id}.tfrecord'
    feature.to_tfrecord(os.path.join(OUTPUT_DATASET, filename))
