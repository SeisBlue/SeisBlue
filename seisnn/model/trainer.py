"""
Training settings.
"""
import os
import shutil

import tensorflow as tf

from seisnn.core import Instance
from seisnn.model.generator import nest_net
import seisnn.example_proto
import seisnn.io
import seisnn.logger
import seisnn.sql
import seisnn.utils

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class BaseTrainer:
    @staticmethod
    def get_dataset_length(database=None):
        count = None
        try:
            db = seisnn.sql.Client(database)
            count = len(db.get_waveform().all())
        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        return count

    @staticmethod
    def get_model_dir(model_instance, remove=False):
        config = seisnn.utils.Config()
        save_model_path = os.path.join(config.models, model_instance)

        if remove:
            shutil.rmtree(save_model_path, ignore_errors=True)
        seisnn.utils.make_dirs(save_model_path)

        save_history_path = os.path.join(save_model_path, "history")
        seisnn.utils.make_dirs(save_history_path)

        return save_model_path, save_history_path


class GeneratorTrainer(BaseTrainer):
    """
    Trainer class.
    """

    def __init__(self, database=None, model=nest_net(),
                 optimizer=tf.keras.optimizers.Adam(1e-4),
                 loss=tf.keras.losses.BinaryCrossentropy()):
        """
        Initialize the trainer.

        :param database: SQL database.
        :param model: keras model.
        :param optimizer: keras optimizer.
        :param loss: keras loss.
        """
        self.database = database
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    @tf.function
    def train_step(self, train, val):
        """
        Training step.

        :param train: Training data.
        :param val: Validation data.
        :rtype: float
        :return: predict loss, validation loss
        """
        with tf.GradientTape(persistent=True) as tape:
            train_pred = self.model(train['trace'], training=True)
            train_loss = self.loss(train['label'], train_pred)

            val_pred = self.model(val['trace'], training=False)
            val_loss = self.loss(val['label'], val_pred)

            gradients = tape.gradient(train_loss,
                                      self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            return train_loss, val_loss

    def train_loop(self,
                   dataset, model_name,
                   epochs=1, batch_size=1,
                   log_step=100, plot=False,
                   remove=False):
        """
        Main training loop.

        :param str dataset: Dataset name.
        :param str model_name: Model directory name.
        :param int epochs: Epoch number.
        :param int batch_size: Batch size.
        :param int log_step: Logging step interval.
        :param bool plot: Plot training validation,
            False save fig, True show fig.
        :param bool remove: If True, remove model folder before training.
        :return:
        """
        model_path, history_path = self.get_model_dir(model_name,
                                                      remove=remove)

        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, model_path,
                                                  max_to_keep=100)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            last_epoch = len(ckpt_manager.checkpoints)
            print(f'Latest checkpoint epoch {last_epoch} restored!!')

        dataset = seisnn.io.read_dataset(dataset).shuffle(100000)
        val = next(iter(dataset.batch(1)))
        metrics_names = ['loss', 'val']

        data_len = self.get_dataset_length(self.database)

        for epoch in range(epochs):
            print(f'epoch {epoch + 1} / {epochs}')

            n = 0
            progbar = tf.keras.utils.Progbar(
                data_len, stateful_metrics=metrics_names)

            loss_buffer = []
            for train in dataset.prefetch(100).batch(batch_size):
                train_loss, val_loss = self.train_step(train, val)
                loss_buffer.append([train_loss, val_loss])

                values = [('loss', train_loss.numpy()),
                          ('val', val_loss.numpy())]
                progbar.add(batch_size, values=values)

                if n % log_step == 0:
                    seisnn.logger.save_loss(loss_buffer, model_name,
                                            model_path)
                    loss_buffer.clear()

                    title = f'epoch{epoch + 1:0>2}_step{n:0>5}___'
                    val['predict'] = self.model.predict(val['trace'])
                    val['id'] = tf.convert_to_tensor(
                        title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

                    example = next(seisnn.example_proto.batch_iterator(val))
                    instance = Instance(example)

                    if plot:
                        instance.plot()
                    else:
                        instance.plot(save_dir=history_path)
                n += 1

            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint to {ckpt_save_path}')

    def export_model(self, model_name):
        model_path, history_path = self.get_model_dir(model_name)
        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, model_path,
                                                  max_to_keep=100)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            last_epoch = len(ckpt_manager.checkpoints)
            print(f'Latest checkpoint epoch {last_epoch} restored!!')

        self.model.save(model_path + ".h5")
