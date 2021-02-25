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

    def score(self, delta=0.1):
        db = seisnn.sql.Client('HL2019.db')
        for phase in ['P', 'S']:
            tp = 0
            predict_pick = db.get_picks(phase=phase, tag='val_predict')
            manual_pick = db.get_picks(phase=phase, tag='val_manual')
            total_predict = len(predict_pick.all())
            total_manual = len(manual_pick.all())
            for pick in predict_pick:
                from_time, to_time = get_from_time_to_time(pick,delta)
                manual = db.get_picks(
                    from_time=str(from_time),
                    to_time=str(to_time),
                    phase = phase,
                    station=pick.station,
                    tag='manual'
                )
                if manual.all():
                    tp = tp + 1
            print(
                f'{phase}: tp = {tp},fp = {total_predict - tp},fn = {total_manual - tp}')
            precision, recall, f1 = seisnn.qc.precision_recall_f1_score(
                true_positive=tp, val_count=total_manual,
                pred_count=total_predict)
            print(
                f'{phase}: precision = {precision},recall = {recall},f1 = {f1}')


def get_from_time_to_time(pick,delta = 0.1):
    from_time = UTCDateTime(pick.time) - delta
    from_time = datetime.strptime(str(from_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    to_time = UTCDateTime(pick.time) + delta
    to_time = datetime.strptime(str(to_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    return from_time, to_time
