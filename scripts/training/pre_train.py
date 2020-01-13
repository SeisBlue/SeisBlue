import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import argparse

import tensorflow as tf

from seisnn.utils import get_config, make_dirs
from seisnn.io import read_dataset

from seisnn.logger import save_loss
from seisnn.model.settings import model, optimizer, train_step
from seisnn.plot import plot_loss
from seisnn.core import Feature
from seisnn.example_proto import batch_iterator

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='dataset', type=str)
ap.add_argument('-m', '--model', required=True, help='save model', type=str)
args = ap.parse_args()

config = get_config()
SAVE_MODEL_PATH = os.path.join(config['MODELS_ROOT'], args.model)
shutil.rmtree(SAVE_MODEL_PATH, ignore_errors=True)

make_dirs(SAVE_MODEL_PATH)
SAVE_HISTORY_PATH = os.path.join(SAVE_MODEL_PATH, "history")
make_dirs(SAVE_HISTORY_PATH)

dataset_dir = os.path.join(config['DATASET_ROOT'], args.dataset)
dataset = read_dataset(dataset_dir).shuffle(10000).take(1000)

val = next(iter(dataset.batch(1)))
val_trace = val['trace']
val_pdf = val['pdf']

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, SAVE_MODEL_PATH, max_to_keep=100)

EPOCHS = 1
for epoch in range(EPOCHS):
    n = 0
    loss_buffer = []
    for train in dataset.prefetch(100).batch(1):
        train_trace = train['trace']
        train_pdf = train['pdf']

        train_loss, val_loss = train_step(train_trace, train_pdf, val_trace, val_pdf)
        loss_buffer.append([train_loss, val_loss])

        if n % 10 == 0:
            print(f'epoch {epoch + 1}, step {n}, loss= {train_loss.numpy():f}, val= {val_loss.numpy():f}')

            pdf = model.predict(val_trace)
            val['pdf'] = tf.concat([val_pdf, pdf], axis=3)

            phase = val['phase'].to_list()
            for p in phase:
                if not [b'p'] in p:
                    p.append([b'p'])

            val['phase'] = tf.ragged.constant(phase)

            title = f'epoch{epoch + 1:0>2}_step{n:0>5}'
            val['id'] = tf.convert_to_tensor(title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

            example = next(batch_iterator(val))
            feature = Feature(example)
            feature.get_picks('p', 'pre_train')
            feature.to_tfrecord(os.path.join(SAVE_HISTORY_PATH, f'{title}.tfrecord'))

            save_loss(loss_buffer, args.model, SAVE_MODEL_PATH)
            loss_buffer.clear()
        n += 1

ckpt_save_path = ckpt_manager.save()
print(f'Saving pre-train checkpoint to {ckpt_save_path}')

feature.plot()
plot_loss(os.path.join(SAVE_MODEL_PATH, f'{args.model}.log'))
