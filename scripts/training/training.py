import os
import glob
import shutil
import argparse

import tensorflow as tf

from seisnn.utils import get_config, make_dirs
from seisnn.io import read_dataset
from seisnn.core import Feature
from seisnn.logger import save_loss
from seisnn.model.settings import model, optimizer, train_step

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='dataset', type=str)
ap.add_argument('-p', '--pre_train', required=True, help='pre-train model', type=str)
ap.add_argument('-m', '--model', required=True, help='save model', type=str)
args = ap.parse_args()

config = get_config()

SAVE_MODEL_PATH = os.path.join(config['MODELS_ROOT'], args.model)
make_dirs(SAVE_MODEL_PATH)
SAVE_HISTORY_PATH = os.path.join(SAVE_MODEL_PATH, "history")
make_dirs(SAVE_HISTORY_PATH)

dataset_dir = os.path.join(config['DATASET_ROOT'], args.dataset)
dataset = read_dataset(dataset_dir).shuffle(10000)

validate = Feature(next(iter(dataset)))
validate.filter_phase('P')
validate.filter_channel('Z')
val_trace = validate.get_trace()
val_pdf = validate.get_pdf()

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, SAVE_MODEL_PATH, max_to_keep=100)

if args.pre_train:
    PRE_TRAIN_PATH = os.path.join(config['MODELS_ROOT'], args.pre_train)
    for file in glob.glob(os.path.join(PRE_TRAIN_PATH,'ckpt*')):
        shutil.copy2(file, SAVE_MODEL_PATH)


if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = len(ckpt_manager.checkpoints)
    print(f'Latest checkpoint epoch {last_epoch} restored!!')

EPOCHS = 1
for epoch in range(EPOCHS):
    n = 0
    loss_buffer = []
    for example in dataset.prefetch(100):
        try:
            feature = Feature(example)
            feature.filter_phase('P')
            feature.filter_channel('Z')

            train_trace = feature.get_trace()
            train_pdf = feature.get_pdf()

            if train_pdf is None or train_trace is None:
                continue

            train_loss, val_loss = train_step(train_trace, train_pdf, val_trace, val_pdf)
            loss_buffer.append([train_loss.numpy(), val_loss.numpy()])

        except:
            continue

        if n % 10 == 0:
            print(f'epoch {epoch + 1}, step {n}, loss= {train_loss.numpy():f}, val= {val_loss.numpy():f}')

        if n % 1000 == 0 and not n == 0:
            validate.phase['pre_train'] = model.predict(val_trace)[0, 0, -3001:, 0]

            title = f'epoch{epoch + 1:0>2}_step{n:0>5}'
            validate.id = title
            validate.to_tfrecord(os.path.join(SAVE_HISTORY_PATH, f'{title}.tfrecord'))

            save_loss(loss_buffer, args.model, SAVE_MODEL_PATH)
            loss_buffer.clear()
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint to {ckpt_save_path}')

        n += 1