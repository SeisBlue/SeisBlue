"""
Evaluator settings.
"""

import os
import shutil

import tensorflow as tf

from seisnn.model.generator import nest_net
from seisnn.data import example_proto, io, logger, sql
from seisnn.data.core import Instance
from seisnn import utils


class BaseEvaluator:
    @staticmethod
    def get_dataset_length(database=None):
        count = None
        try:
            db = sql.Client(database)
            count = len(db.get_waveform().all())
        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        return count

    @staticmethod
    def get_model_dir(model_instance):
        config = utils.get_config()
        save_model_path = os.path.join(config['MODELS_ROOT'], model_instance)
        save_history_path = os.path.join(save_model_path, "history")
        utils.make_dirs(save_history_path)

        return save_model_path, save_history_path


class GeneratorEvaluator(BaseEvaluator):
    """
    Trainer class.
    """

    def __init__(self,
                 database=None,
                 model_name=None):
        """
        Initialize the evaluator.

        :param database: SQL database.
        :param model_name: Saved model.
        """
        self.database = database
        self.model_name = model_name

    def eval(self, dataset):
        """
        Main eval loop.


        :param str dataset: Dataset name.
        :param str model_name: Model directory name.

        """
        model_path, history_path = self.get_model_dir(self.model_name)
        dataset = io.read_dataset(dataset)
        val = next(iter(dataset.batch(1)))

        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, model_path,
                                                  max_to_keep=100)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            last_epoch = len(ckpt_manager.checkpoints)
            print(f'Latest checkpoint epoch {last_epoch} restored!!')

        metrics_names = ['loss', 'val']

        data_len = self.get_dataset_length(self.database)
        progbar = tf.keras.utils.Progbar(
            data_len, stateful_metrics=metrics_names)

        for epoch in range(epochs):
            print(f'epoch {epoch + 1} / {epochs}')
            n = 0
            loss_buffer = []
            for train in dataset.prefetch(100).batch(batch_size):
                train_loss, val_loss = self.train_step(train, val)
                loss_buffer.append([train_loss, val_loss])

                values = [('loss', train_loss.numpy()),
                          ('val', val_loss.numpy())]
                progbar.add(batch_size, values=values)

                if n % log_step == 0:
                    logger.save_loss(loss_buffer, model_name, model_path)
                    loss_buffer.clear()

                    title = f'epoch{epoch + 1:0>2}_step{n:0>5}___'
                    val['predict'] = self.model.predict(val['trace'])
                    val['id'] = tf.convert_to_tensor(
                        title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

                    example = next(example_proto.batch_iterator(val))
                    instance = Instance(example)

                    if plot:
                        instance.plot()
                    else:
                        instance.plot(save_dir=history_path)
                n += 1

        ckpt_save_path = ckpt_manager.save()
        print(f'Saving pre-train checkpoint to {ckpt_save_path}')
