"""
Training step settings.
"""
import os
import shutil

import tensorflow as tf

from seisnn.model.unet import nest_net
from seisnn.data import example_proto, io, logger
from seisnn.data.core import Instance
from seisnn import utils


class Trainer:
    """
    Trainer class.
    """
    def __init__(self,
                 model=nest_net(color_type=1, num_class=3),
                 optimizer=tf.keras.optimizers.Adam(1e-4),
                 loss=tf.keras.losses.BinaryCrossentropy()):
        """
        Initialize the trainer.

        :param model: keras model.
        :param optimizer: keras optimizer.
        :param loss: keras loss.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    @tf.function
    def train_step(self, train_trace, train_label, val_trace, val_label):
        """
        Training step.

        :param train_trace: Training trace data.
        :param train_label: Training trace label.
        :param val_trace: Validation trace data.
        :param val_label: Validation trace label.
        :rtype: float
        :return: (predict loss, validation loss)
        """
        with tf.GradientTape(persistent=True) as tape:
            pred_label = self.model(train_trace, training=True)
            pred_loss = self.loss(train_label, pred_label)

            pred_val_label = self.model(val_trace, training=False)
            val_loss = self.loss(val_label, pred_val_label)

            gradients = tape.gradient(pred_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            return pred_loss, val_loss

    def train_loop(self, dataset, model_instance):
        """
        Main training loop.

        :param dataset: Dataset name.
        :param model_instance: Model name.
        :return:
        """
        config = utils.get_config()
        SAVE_MODEL_PATH = os.path.join(config['MODELS_ROOT'], model_instance)
        shutil.rmtree(SAVE_MODEL_PATH, ignore_errors=True)

        utils.make_dirs(SAVE_MODEL_PATH)
        SAVE_HISTORY_PATH = os.path.join(SAVE_MODEL_PATH, "history")
        utils.make_dirs(SAVE_HISTORY_PATH)

        dataset = io.read_dataset(dataset).shuffle(100000)

        val = next(iter(dataset.batch(1)))
        val_trace = val['trace']
        val_label = val['label']

        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, SAVE_MODEL_PATH,
                                                  max_to_keep=100)

        EPOCHS = 10
        for epoch in range(EPOCHS):
            n = 0
            loss_buffer = []
            for train in dataset.prefetch(100).batch(1):
                train_trace = train['trace']
                train_label = train['label']

                train_loss, val_loss = self.train_step(train_trace, train_label,
                                                  val_trace, val_label)
                loss_buffer.append([train_loss, val_loss])

                if n % 100 == 0:
                    print(f'epoch {epoch + 1}, step {n}, '
                          f'loss= {train_loss.numpy():f},'
                          f' val= {val_loss.numpy():f}')

                    val['predict'] = self.model.predict(val_trace)

                    phase = val['phase'].to_list()

                    val['phase'] = tf.ragged.constant(phase)

                    title = f'epoch{epoch + 1:0>2}_step{n:0>5}'
                    val['id'] = \
                        tf.convert_to_tensor(title.encode('utf-8'),
                                             dtype=tf.string)[
                            tf.newaxis]

                    example = next(example_proto.batch_iterator(val))
                    instance = Instance(example)

                    instance.to_tfrecord(
                        os.path.join(SAVE_HISTORY_PATH, f'{title}.tfrecord'))

                    logger.save_loss(loss_buffer, model_instance,
                                     SAVE_MODEL_PATH)
                    loss_buffer.clear()
                    instance.plot()
                n += 1

        ckpt_save_path = ckpt_manager.save()
        print(f'Saving pre-train checkpoint to {ckpt_save_path}')


if __name__ == "__main__":
    pass
