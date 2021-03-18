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

    def eval(self, tfr_list, batch_size=100):
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

        data_len = self.get_dataset_length(self.database)
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

    def score(self, delta=0.1, error_distribution=True):
        db = seisnn.sql.Client('HL2019.db')
        for phase in ['P', 'S']:
            tp = 0
            error = []
            predict_pick = db.get_picks(phase=phase, tag='val_predict')
            label_pick = db.get_picks(phase=phase, tag='val_label')
            total_predict = len(predict_pick.all())
            total_label = len(label_pick.all())
            print(f'{phase}_total_predict: {total_predict} '
                  f'{phase}_total_label: {total_label}')

            for pick in predict_pick:
                from_time, to_time = get_from_time_to_time(pick, delta)
                label = db.get_picks(
                    from_time=str(from_time),
                    to_time=str(to_time),
                    phase=phase,
                    station=pick.station,
                    tag='val_label'
                )
                if label.all():
                    tp = tp + 1
                    if error_distribution:
                        error.append(UTCDateTime(label[0].time) - UTCDateTime(
                            pick.time))
            plot_error_distribution(error)
            print(
                f'{phase}: tp = {tp},fp = {total_predict - tp},fn = {total_label - tp}')
            precision, recall, f1 = seisnn.qc.precision_recall_f1_score(
                true_positive=tp, val_count=total_label,
                pred_count=total_predict)
            print(
                f'{phase}: precision = {precision},recall = {recall},f1 = {f1}')


def get_from_time_to_time(pick, delta=0.1):
    from_time = UTCDateTime(pick.time) - delta
    from_time = datetime.strptime(str(from_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    to_time = UTCDateTime(pick.time) + delta
    to_time = datetime.strptime(str(to_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    return from_time, to_time
