"""
Training settings.
"""
import os
import shutil

import tensorflow as tf

from seisblue.core import Instance
from seisblue.model.generator import nest_net
import seisblue.example_proto
import seisblue.io
import seisblue.logger
import seisblue.sql
import seisblue.utils

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class BaseTrainer:
    @staticmethod
    def get_dataset_length(database=None, tfr_list=None):
        count = None
        try:
            db = seisblue.sql.Client(database)
            tfr_list = seisblue.utils.flatten_list(tfr_list)
            counts = db.get_tfrecord(path=tfr_list, column='count')
            count = sum(seisblue.utils.flatten_list(counts))
        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        return count

    @staticmethod
    def get_model_dir(model_instance, remove=False):
        config = seisblue.utils.Config()
        save_model_path = os.path.join(config.models, model_instance)

        if remove:
            shutil.rmtree(save_model_path, ignore_errors=True)
        seisblue.utils.make_dirs(save_model_path)

        save_history_path = os.path.join(save_model_path, "history")
        seisblue.utils.make_dirs(save_history_path)

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

        self.train_acc_metric = tf.keras.metrics.BinaryCrossentropy()
        self.val_acc_metric = tf.keras.metrics.BinaryCrossentropy()

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

    @tf.function
    def train_step(self, train):
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

        gradients = tape.gradient(train_loss,
                                  self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_acc_metric.update_state(train['label'], train_pred)
        return train_loss

    @tf.function
    def val_step(self, val):
        val_pred = self.model(val['trace'], training=False)
        self.val_acc_metric.update_state(val['label'], val_pred)

    def train_loop(self,
                   tfr_list, model_name,
                   epochs=1, batch_size=1,
                   log_step=100,
                   plot=False,
                   save_figure=False,
                   remove=False):
        """
        Main training loop.

        :param tfr_list: List of TFRecord path.
        :param str model_name: Model directory name.
        :param int epochs: Epoch number.
        :param int batch_size: Batch size.
        :param int log_step: Logging step interval.
        :param bool plot: Plot training validation,
            False save fig, True show fig.
        :param bool save_figure: if True, save fig else not
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

        dataset = tf.data.TFRecordDataset(tfr_list)

        dataset_train, dataset_validation = self.train_val_split(dataset, 9)

        dataset_train = seisblue.io.parse_dataset(dataset_train)
        train_batch_size = len(list(dataset_train))

        dataset_validation = seisblue.io.parse_dataset(dataset_validation)
        val_batch_size = len(list(dataset_validation))

        plot_data = next(iter(dataset_validation.batch(1)))

        metrics_names = ['train_loss']

        for epoch in range(epochs):
            print(f'epoch {epoch + 1} / {epochs}')
            progbar = tf.keras.utils.Progbar(
                train_batch_size, stateful_metrics=metrics_names)

            for val_from_dataset in dataset_validation.prefetch(100) \
                    .batch(val_batch_size):
                val = val_from_dataset
            for n, train in enumerate(dataset_train.prefetch(100).shuffle(50000) \
                                              .batch(batch_size)):

                train_loss = self.train_step(train)

                values = [('train_loss', train_loss.numpy())]

                progbar.add(len(train['id']), values=values)

                if n % log_step == 0:
                    if plot or save_figure:
                        title = f'epoch{epoch + 1:0>2}_step{n:0>5}___'
                        plot_data['predict'] = self.model.predict(
                            plot_data['trace'])
                        plot_data['id'] = tf.convert_to_tensor(
                            title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

                        example = next(
                            seisblue.example_proto.batch_iterator(plot_data))
                        instance = Instance(example)

                        if plot:
                            instance.plot()
                        elif save_figure:
                            instance.plot(save_dir=history_path)

            self.val_step(val)

            print(
                f'train_loss = {float(self.train_acc_metric.result()):.3f}'
                f', val_loss = {float(self.val_acc_metric.result()):.3f} in epoch')

            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()

            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint to {ckpt_save_path}')
            config = seisblue.utils.Config()
            self.model.save(f'{os.path.join(config.models, model_name)}.h5')

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

    @staticmethod
    def train_val_split(dataset, split_num):
        split_num = split_num
        dataset_train = dataset \
            .window(split_num, split_num + 1) \
            .flat_map(lambda ds: ds)
        dataset_validation = dataset \
            .skip(split_num) \
            .window(1, split_num + 1) \
            .flat_map(lambda ds: ds)
        return dataset_train, dataset_validation
