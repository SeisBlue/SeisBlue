"""
Evaluator settings.
"""

import os

import numpy as np
import obspy
import tensorflow as tf
from obspy.core.utcdatetime import UTCDateTime
from datetime import datetime
import time
import seisblue
from seisblue.core import Instance
from seisblue.model.attention import TransformerBlockE, MultiHeadSelfAttention, \
    ResBlock
from seisblue.plot import plot_error_distribution
import seisblue.example_proto
import seisblue.io
import seisblue.sql
import seisblue.utils


class BaseEvaluator:
    database = None

    def get_dataset_length(self):
        count = None
        try:
            db = seisblue.sql.Client(self.database)
            count = len(db.get_waveform())
        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        return count

    @staticmethod
    def get_model_dir(model_instance):
        config = seisblue.utils.Config()
        save_model_path = os.path.join(config.models, model_instance)
        return save_model_path

    @staticmethod
    def get_eval_dir(dir_name):
        config = seisblue.utils.Config()
        eval_path = os.path.join(config.eval, dir_name.split('.')[0])
        seisblue.utils.make_dirs(eval_path)

        return eval_path


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

    def predict(self, tfr_list, batch_size=500):
        """
        Main eval loop.

        :param tfr_list: List of .tfrecord.
        :param int batch_size: batch size.
        """
        model_path = self.get_model_dir(self.model_name)

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TransformerBlockE': TransformerBlockE,
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'ResBlock': ResBlock
            })

        dataset = seisblue.io.read_dataset(tfr_list)
        n = 0
        for val in dataset.prefetch(100).batch(batch_size):
            progbar = tf.keras.utils.Progbar(len(val['label']))
            val['predict'] = self.model.predict(val['trace'])
            iterator = seisblue.example_proto.batch_iterator(val)
            for i in range(len(val['predict'])):
                config = seisblue.utils.Config()
                instance = Instance(next(iterator))

                sub_dir = getattr(config, 'eval')
                sub_dir = os.path.join(sub_dir, self.model_name)

                file_name = instance.get_tfrecord_name()
                net, sta, loc, chan, year, julday, suffix = file_name.split(
                    '.')
                tfr_dir = os.path.join(sub_dir, year, net, sta)
                seisblue.utils.make_dirs(tfr_dir)
                save_file = os.path.join(tfr_dir, f'{n:0>6}.' + file_name)
                instance.to_tfrecord(save_file)
                progbar.add(1)
                n = n + 1

    def score(self, tfr_list, delta=0.1, threshold=0.5,
              error_distribution=True):
        dataset = seisblue.io.read_dataset(tfr_list)
        progbar = tf.keras.utils.Progbar(len(tfr_list))
        error_array = [[], []]
        true_positive = [0, 0]
        num_of_predict = [0, 0]
        num_of_label = [0, 0]
        for val in dataset.prefetch(100):
            instance = Instance(val)
            instance.label.get_picks(threshold=threshold)
            instance.predict.get_picks(threshold=threshold)
            for pick in instance.label.picks:
                for i, phase in enumerate(['P', 'S']):
                    true_positive[i], error_array[i], num_of_label[i] = \
                        self.judge(pick,
                                   instance,
                                   delta,
                                   true_positive[i],
                                   error_array[i],
                                   num_of_label[i],
                                   phase=phase)

            for pick in instance.predict.picks:
                if pick.phase == 'P':
                    num_of_predict[0] += 1
                if pick.phase == 'S':
                    num_of_predict[1] += 1
            progbar.add(1)
        for i, phase in enumerate(['P', 'S']):
            precision, recall, f1 = seisblue.qc.precision_recall_f1_score(
                true_positive=true_positive[i],
                val_count=num_of_label[i],
                pred_count=num_of_predict[i])
            if error_distribution:
                plot_error_distribution(error_array[i])
            print(f'num_{phase}_predict = {num_of_predict[i]}')
            print(f'num_{phase}_label = {num_of_label[i]}')
            print(
                f'{phase}: precision = {precision},recall = {recall},f1 = {f1}')

    def judge(self, pick, instance, delta, num_true_positive, error_array,
              num_of_label, phase='P'):
        if pick.phase == phase:
            for p_pick in instance.predict.picks:
                if p_pick.phase == pick.phase:
                    if pick.time - delta <= p_pick.time <= pick.time + delta:
                        num_true_positive = num_true_positive + 1
                        error_array.append(
                            p_pick.time - pick.time)
            num_of_label += 1
        return num_true_positive, error_array, num_of_label



