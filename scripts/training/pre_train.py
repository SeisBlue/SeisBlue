import os
import argparse

import tensorflow as tf

from seisnn.utils import get_config, make_dirs
from seisnn.io import read_dataset
from seisnn.feature import Feature
from seisnn.model.settings import model, optimizer, train_step

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='dataset', type=str)
ap.add_argument('-m', '--model', required=True, help='save model', type=str)
args = ap.parse_args()

config = get_config()
SAVE_MODEL_PATH = os.path.join(config['MODELS_ROOT'], args.model)
make_dirs(SAVE_MODEL_PATH)

dataset = read_dataset(args.dataset)
test = Feature(next(iter(read_dataset(args.dataset))))
test.filter_phase('P')
test.filter_channel('Z')

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, SAVE_MODEL_PATH, max_to_keep=100)

EPOCHS = 100
for epoch in range(EPOCHS):
    n = 0
    for example in dataset:
        feature = Feature(example)
        feature.filter_phase('P')
        feature.filter_channel('Z')

        trace = feature.get_trace()
        pdf = feature.get_pdf()

        if pdf is None or trace is None:
            continue

        train_loss = train_step(trace, pdf)
        n += 1

        if n % 100 == 0:
            print(f'epoch {epoch + 1}, step {n}, loss= {train_loss.numpy()}')

    test.phase['pred'] = model.predict(test.get_trace())[0, 0, :3001, 0]
    test.plot()

    ckpt_save_path = ckpt_manager.save()
