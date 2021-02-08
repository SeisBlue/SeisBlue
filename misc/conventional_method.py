"""
Evaluator settings.
"""

import os

import tensorflow as tf
import numpy as np
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import ar_pick

from seisnn.core import Instance
from seisnn.model.attention import TransformerBlockE, TransformerBlockD, \
    MultiHeadSelfAttention, ResBlock
import seisnn.example_proto
import seisnn.io
import seisnn.sql
import seisnn.utils


class BaseEvaluator:
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
    def get_model_dir(model_instance):
        config = seisnn.utils.get_config()
        save_model_path = os.path.join(config['MODELS_ROOT'], model_instance)
        return save_model_path

    @staticmethod
    def get_eval_dir(dataset):
        config = seisnn.utils.get_config()
        dataset_path = os.path.join(config['DATASET_ROOT'], dataset)
        eval_path = os.path.join(config['DATASET_ROOT'], "eval")
        seisnn.utils.make_dirs(eval_path)

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

        dataset = seisnn.io.read_dataset(dataset)
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

            example = next(seisnn.example_proto.batch_iterator(val))
            instance = Instance(example)
            instance.to_tfrecord(os.path.join(eval_path, title + '.tfrecord'))
            n += 1

    def STA_LTA(self, dataset, batch_size=100, short_window=30,
                long_window=200):
        """
        STA/LTA method

        :param dataset: Dataset name.
        :param batch_size: Model directory name.
        """

        dataset_path, eval_path = self.get_eval_dir(dataset)

        dataset = seisnn.io.read_dataset(dataset)
        data_len = self.get_dataset_length(self.database)
        progbar = tf.keras.utils.Progbar(data_len)

        n = 0
        for val in dataset.prefetch(100).batch(batch_size):
            progbar.add(batch_size)
            title = f"eval_{n:0>5}"
            trace_len = val['trace'].shape[2]
            batch_len = val['trace'].shape[0]
            predict = np.zeros((batch_len, 1, 3008, 3))

            for i in range(batch_len):
                z_trace = val['trace'].numpy()[i, :, :, 0].reshape(trace_len)
                cft = recursive_sta_lta(z_trace, short_window, long_window)
                predict[i, :, :, 0] = cft
            val['predict'] = predict
            val['id'] = tf.convert_to_tensor(
                title.encode('utf-8'), dtype=tf.string)[tf.newaxis]
            example = next(seisnn.example_proto.batch_iterator(val))
            instance = Instance(example)
            instance.to_tfrecord(os.path.join(eval_path, title + '.tfrecord'))
            n += 1

    def AR_AIC(self, dataset, batch_size=100):
        """
        AR_AIC method
        :param dataset: Dataset name.
        :param batch_size: Model directory name.
        :return:
        """
        dataset_path, eval_path = self.get_eval_dir(dataset)

        dataset = seisnn.io.read_dataset(dataset)
        data_len = self.get_dataset_length(self.database)
        progbar = tf.keras.utils.Progbar(data_len)
        df = 100
        n = 0
        for val in dataset.prefetch(100).batch(batch_size):
            progbar.add(batch_size)
            title = f"eval_{n:0>5}"
            trace_len = val['trace'].shape[2]
            batch_len = val['trace'].shape[0]
            for i in range(batch_len):
                x_z = val['trace'].numpy()[i, :, :, 0]
                x_n = val['trace'].numpy()[i, :, :, 1]
                x_e = val['trace'].numpy()[i, :, :, 2]
                p_pick, s_pick = ar_pick(
                    a=x_z, b=x_n, c=x_e,
                    samp_rate=df,
                    f1=1.0, f2=45,
                    lta_p=2, sta_p=0.3,
                    lta_s=2, sta_s=0.3,
                    m_p=2, m_s=8,
                    l_p=0.1, l_s=0.2
                )
