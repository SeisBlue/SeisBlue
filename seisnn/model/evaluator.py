"""
Evaluator settings.
"""

import os

import tensorflow as tf

from seisnn.data import example_proto, io, sql
from seisnn.data.core import Instance
from seisnn import utils
from seisnn.model.attention import TransformerBlockE, TransformerBlockD, \
    MultiHeadSelfAttention, ResBlock


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
        return save_model_path

    @staticmethod
    def get_eval_dir(dataset):
        config = utils.get_config()
        dataset_path = os.path.join(config['DATASET_ROOT'], dataset)
        eval_path = os.path.join(config['DATASET_ROOT'], "eval")
        utils.make_dirs(eval_path)

        return dataset_path, eval_path


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
        self.model = None

    def eval(self, dataset, batch_size=100):
        """
        Main eval loop.


        :param str dataset: Dataset name.
        :param str model_name: Model directory name.

        """
        model_path = self.get_model_dir(self.model_name)
        dataset_path, eval_path = self.get_eval_dir(dataset)

        dataset = io.read_dataset(dataset)
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TransformerBlockE': TransformerBlockE,
                'TransformerBlockD': TransformerBlockD,
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'ResBlock': ResBlock
            })

        data_len = self.get_dataset_length(self.database)
        progbar = tf.keras.utils.Progbar(data_len)

        n = 0
        for val in dataset.prefetch(100).batch(batch_size):
            progbar.add(batch_size)

            title = f"eval_{n:0>5}"
            val['predict'] = self.model.predict(val['trace'])
            val['id'] = tf.convert_to_tensor(
                title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

            example = next(example_proto.batch_iterator(val))
            instance = Instance(example)
            instance.to_tfrecord(os.path.join(eval_path, title + '.tfrecord'))
            n += 1
