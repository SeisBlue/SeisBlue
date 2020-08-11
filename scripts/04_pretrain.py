import os
import shutil

import tensorflow as tf

from seisnn.data import example_proto, io, logger
from seisnn.data.core import Instance
from seisnn.model.settings import model, optimizer, train_step
from seisnn.plot import plot_loss
from seisnn.utils import get_config, make_dirs

dataset = 'HL2017'
model_instance = 'test_model'

config = get_config()
SAVE_MODEL_PATH = os.path.join(config['MODELS_ROOT'], model_instance)
shutil.rmtree(SAVE_MODEL_PATH, ignore_errors=True)

make_dirs(SAVE_MODEL_PATH)
SAVE_HISTORY_PATH = os.path.join(SAVE_MODEL_PATH, "history")
make_dirs(SAVE_HISTORY_PATH)

dataset = io.read_dataset(dataset).shuffle(100000)

val = next(iter(dataset.batch(1)))
val_trace = val['trace']
val_label = val['label']

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, SAVE_MODEL_PATH,
                                          max_to_keep=100)

EPOCHS = 10
for epoch in range(EPOCHS):
    n = 0
    loss_buffer = []
    for train in dataset.prefetch(100).batch(1):
        train_trace = train['trace']
        train_label = train['label']

        train_loss, val_loss = train_step(train_trace, train_label,
                                          val_trace, val_label)
        loss_buffer.append([train_loss, val_loss])

        if n % 100 == 0:
            print(f'epoch {epoch + 1}, step {n}, '
                  f'loss= {train_loss.numpy():f},'
                  f' val= {val_loss.numpy():f}')

            val['predict'] = model.predict(val_trace)

            phase = val['phase'].to_list()

            val['phase'] = tf.ragged.constant(phase)

            title = f'epoch{epoch + 1:0>2}_step{n:0>5}'
            val['id'] = \
                tf.convert_to_tensor(title.encode('utf-8'), dtype=tf.string)[
                    tf.newaxis]

            example = next(example_proto.batch_iterator(val))
            instance = Instance(example)

            instance.to_tfrecord(
                os.path.join(SAVE_HISTORY_PATH, f'{title}.tfrecord'))

            logger.save_loss(loss_buffer, model_instance, SAVE_MODEL_PATH)
            loss_buffer.clear()
            instance.plot()
        n += 1

ckpt_save_path = ckpt_manager.save()
print(f'Saving pre-train checkpoint to {ckpt_save_path}')

plot_loss(os.path.join(SAVE_MODEL_PATH, f'{model_instance}.log'))
