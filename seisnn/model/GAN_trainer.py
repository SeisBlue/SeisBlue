import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate

from seisnn.core import Instance
from seisnn.model.generator import nest_net
from seisnn.model.attention import transformer
from seisnn.model.GAN_model import build_discriminator, build_cgan
import seisnn.example_proto
import seisnn.io
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
        save_model_path = os.path.join(config['MODELS_ROOT'], model_instance)

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
        self.generator_model = transformer(img_rows=1, img_cols=3008,
                                           color_type=3,
                                           num_class=3)
        self.discriminator_model = build_discriminator(img_rows=1,
                                                       img_cols=3008,
                                                       color_type=3,
                                                       num_class=3)
        self.generator_optimizer = Adam(learning_rate=1e-3)
        self.generator_model.compile(loss='binary_crossentropy',
                                     optimizer=self.generator_optimizer)

        self.discriminator_model.trainable = False

        self.cgan_model = build_cgan(self.generator_model,
                                     self.discriminator_model)

        loss = [tf.keras.losses.BinaryCrossentropy(),
                tf.keras.losses.BinaryCrossentropy()]
        loss_weights = [100000, 1]
        self.cgan_optimizer = Adam(learning_rate=1e-3)
        self.cgan_model.compile(loss=loss, loss_weights=loss_weights,
                                optimizer=self.cgan_optimizer)
        self.discriminator_optimizer = Adam(learning_rate=1e-3)
        self.discriminator_model.trainable = True
        self.discriminator_model.compile(loss='binary_crossentropy',
                                         optimizer=self.discriminator_optimizer)

        self.database = database
        self.model = self.cgan_model
        self.optimizer = optimizer

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

        ckpt = tf.train.Checkpoint(
            generator_model=self.generator_model,
            discriminator_model=self.discriminator_model,
            cgan_model=self.cgan_model,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            cgan_optimizer=self.cgan_optimizer,
        )
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

            for train in dataset.prefetch(100).batch(batch_size):
                d_loss, g_loss = self.train_step(train, val)
                values = [('d_loss', d_loss),
                          ('g_loss', g_loss)]
                progbar.add(batch_size, values=values)

                n += 1
                if n % log_step == 0:
                    val['predict'] = self.generator_model(val['trace'])
                    concate = concatenate([val['trace'], val['predict']],
                                          axis=3)
                    score = self.discriminator_model(concate)
                    print(score)
                    title = f'epoch{epoch + 1:0>2}_step{n:0>5}___'

                    val['id'] = tf.convert_to_tensor(
                        title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

                    example = next(seisnn.example_proto.batch_iterator(val))
                    instance = Instance(example)

                    if plot:
                        instance.plot()
                    else:
                        instance.plot(save_dir=history_path)

            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint to {ckpt_save_path}')
            self.generator_model.save('/home/andy/models/test_model.h5')

    def train_step(self, train, val):
        """
        Training step.

        :param train: Training data.
        :param val: Validation data.
        :rtype: float
        :return: predict loss, validation loss
        """

        real = np.ones((train['trace'].shape[0], 1))
        fake = np.zeros((train['trace'].shape[0], 1))
        g_pred = self.generator_model(train['trace'], training=False)
        concat = concatenate((train['trace'], g_pred), axis=3)
        f_disc_loss = self.discriminator_model.train_on_batch(concat, fake)
        concat = concatenate((train['trace'], train['label']), axis=3)
        r_disc_loss = self.discriminator_model.train_on_batch(concat, real)
        disc_loss = 0.5 * (f_disc_loss + r_disc_loss)
        self.discriminator_model.trainable = False
        gen_loss = self.cgan_model.train_on_batch(train['trace'],
                                                  [train['label'], real])
        self.discriminator_model.trainable = True

        return np.array(disc_loss), np.array(gen_loss)


if __name__ == "__main__":
    dataset = 'HL2019'
    model_instance = 'test_model'
    database = 'HL2019.db'

    trainer = GeneratorTrainer(database)
    trainer.train_loop(dataset, model_instance, plot=True, batch_size=500,
                       remove=True, epochs=500, log_step=10)
