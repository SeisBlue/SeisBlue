"""
Evaluator settings.
"""

import os

import tensorflow as tf
from obspy.core.utcdatetime import UTCDateTime
from datetime import datetime

from seisnn.core import Instance
from seisnn.model.attention import TransformerBlockE, TransformerBlockD, \
    MultiHeadSelfAttention, ResBlock
from seisnn.plot import plot_error_distribution
import seisnn.example_proto
import seisnn.io
import seisnn.sql
import seisnn.utils


class BaseEvaluator:
    database = None
    def get_dataset_length(self):
        count = None
        try:
            db = seisnn.sql.Client(self.database)
            count = len(db.get_waveform().all())
        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        return count

    @staticmethod
    def get_model_dir(model_instance):
        config = seisnn.utils.Config()
        save_model_path = os.path.join(config.models, model_instance)
        return save_model_path

    @staticmethod
    def get_eval_dir(dir_name):
        config = seisnn.utils.Config()
        eval_path = os.path.join(config.eval, dir_name.split('.')[0])
        seisnn.utils.make_dirs(eval_path)

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

    def get_eval_list(self):
        eval_path = self.get_eval_dir(self.model_name)
        eval_list = []
        for _, _, files in os.walk(eval_path):
            for file in files:
                eval_list.append(os.path.join(eval_path, file))
        return eval_list

    def predict(self, tfr_list, batch_size=500):
        """
        Main eval loop.

        :param tfr_list: List of .tfrecord.
        :param str name: Output name.
        """
        model_path = self.get_model_dir(self.model_name)
        eval_path = self.get_eval_dir(self.model_name)

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TransformerBlockE': TransformerBlockE,
                'TransformerBlockD': TransformerBlockD,
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'ResBlock': ResBlock
            })

        data_len = self.get_dataset_length()
        progbar = tf.keras.utils.Progbar(data_len)
        dataset = seisnn.io.read_dataset(tfr_list)
        n = 0
        for val in dataset.prefetch(100).batch(batch_size):
            progbar.add(batch_size)

            val['predict'] = self.model.predict(val['trace'])

            iterator = seisnn.example_proto.batch_iterator(val)
            for i in range(len(val['predict'])):
                title = f"eval_{n:0>5}"

                instance = Instance(next(iterator))
                instance.to_tfrecord(
                    os.path.join(eval_path, title + '.tfrecord'))
                n += 1

            val['id'] = tf.convert_to_tensor(
                title.encode('utf-8'), dtype=tf.string)[tf.newaxis]

            example = next(seisnn.example_proto.batch_iterator(val))
            instance = Instance(example)
            instance.to_tfrecord(os.path.join(eval_path, title + '.tfrecord'))
            n += 1


    def score(self, tfr_list, batch_size=500, delta=0.1,
               error_distribution=True):
        P_true_positive = 0
        S_true_positive = 0
        P_error_array = []
        S_error_array = []
        num_P_predict = 0
        num_S_predict = 0
        num_P_label = 0
        num_S_label = 0
        dataset = seisnn.io.read_dataset(tfr_list)
        for val in dataset.prefetch(100).batch(batch_size):
            iterator = seisnn.example_proto.batch_iterator(val)
            progbar = tf.keras.utils.Progbar(len(val['predict']))
            for i in range(len(val['predict'])):
                instance = Instance(next(iterator))
                instance.label.get_picks()
                instance.predict.get_picks()

                for pick in instance.label.picks:
                    if pick.phase == 'P':
                        for p_pick in instance.predict.picks:
                            if p_pick.phase==pick.phase:
                                if pick.time-delta<=p_pick.time<=pick.time+delta:
                                    P_true_positive = P_true_positive+1
                                    P_error_array.append(p_pick.time-pick.time)

                        num_P_label +=1
                    if pick.phase == 'S':
                        for p_pick in instance.predict.picks:
                            if p_pick.phase==pick.phase:
                                if pick.time-delta<=p_pick.time<=pick.time+delta:
                                    S_true_positive = S_true_positive+1
                                    S_error_array.append(p_pick.time - pick.time)
                        num_S_label += 1
                for pick in instance.predict.picks:
                    if pick.phase == 'P':
                        num_P_predict +=1
                    if pick.phase == 'S':
                        num_S_predict += 1
                progbar.add(1)
        for phase in ['P','S']:
            precision, recall, f1 = seisnn.qc.precision_recall_f1_score(
                true_positive=eval(f'{phase}_true_positive'), val_count=eval(f'num_{phase}_label'),
                pred_count=eval(f'num_{phase}_predict'))
            if error_distribution:
                plot_error_distribution(eval(f'{phase}_error_array'))
            print(
                f'{phase}: precision = {precision},recall = {recall},f1 = {f1}')


def get_from_time_to_time(pick, delta=0.1):
    from_time = UTCDateTime(pick.time) - delta
    from_time = datetime.strptime(str(from_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    to_time = UTCDateTime(pick.time) + delta
    to_time = datetime.strptime(str(to_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    return from_time, to_time
