import os
import argparse

import tensorflow as tf

from seisnn.utils import get_config, make_dirs
from seisnn.io import read_dataset
from seisnn.core import Feature
from seisnn.logger import save_history
from seisnn.model.settings import model, optimizer, train_step

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='dataset', type=str)
ap.add_argument('-m', '--model', required=True, help='save model', type=str)
args = ap.parse_args()

config = get_config()
SAVE_MODEL_PATH = os.path.join(config['MODELS_ROOT'], args.model)
make_dirs(SAVE_MODEL_PATH)

SAVE_LOG_PATH = os.path.join(SAVE_MODEL_PATH, "log")
make_dirs(SAVE_LOG_PATH)

dataset_dir = os.path.join(config['DATASET_ROOT'], args.dataset)
dataset = read_dataset(dataset_dir).shuffle(10000).take(1000)
test = Feature(next(iter(dataset)))
test.filter_phase('P')
test.filter_channel('Z')

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, SAVE_MODEL_PATH, max_to_keep=100)

EPOCHS = 5
for epoch in range(EPOCHS):
    n = 0
    for example in dataset.prefetch(100):
        try:
            feature = Feature(example)
            feature.filter_phase('P')
            feature.filter_channel('Z')

            trace = feature.get_trace()
            pdf = feature.get_pdf()

            if pdf is None or trace is None:
                continue

            train_loss = train_step(trace, pdf)

        except:
            continue

        if n % 10 == 0:
            print(f'epoch {epoch + 1}, step {n}, loss= {train_loss.numpy()}')
            test.phase['pre_train'] = model.predict(test.get_trace())[0, 0, -3001:, 0]

            title = f'epoch{epoch + 1:0>2}_step{n:0>5}'
            test.id = title
            save_dir = os.path.join(SAVE_LOG_PATH, 'pre_train')
            save_history(test, title, save_dir)

        n += 1

ckpt_save_path = ckpt_manager.save()
print(f'Saving pre-train checkpoint to {ckpt_save_path}')
test.plot()