"""
Example Protocol
"""
import numpy as np
import tensorflow as tf


class Feature:
    __slots__ = [
        'id',
        'station',
        'starttime',
        'endtime',

        'npts',
        'delta',

        'trace',
        'channel',

        'phase',
        'label',
        'predict',
    ]


def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    Returns a float_list from a float / double.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def feature_to_example(feature):
    """
    Returns tf.SequenceExample serialize string from feature dict.

    :param Feature feature: Feature dict extract from stream.
    :return: Serialized example.
    """
    # Convert array data into numpy bytes string.
    for key in ['trace', 'label', 'predict']:
        if isinstance(getattr(feature, key), tf.Tensor):
            setattr(feature, key, getattr(feature, key).numpy())

    # Convert single data into tf.train.Features.
    context_data = {
        'id': _bytes_feature(feature.id),
        'starttime': _bytes_feature(feature.starttime),
        'endtime': _bytes_feature(feature.endtime),
        'station': _bytes_feature(feature.station),

        'npts': _int64_feature(feature.npts),
        'delta': _float_feature(feature.delta),

        'trace': _bytes_feature(
            feature.trace.astype(dtype=np.float32).tostring()),
        'label': _bytes_feature(
            feature.label.astype(dtype=np.float32).tostring()),
        'predict': _bytes_feature(
            feature.predict.astype(dtype=np.float32).tostring()),
    }
    context = tf.train.Features(feature=context_data)

    # Convert list of tf.train.Feature into tf.train.FeatureLists.
    sequence_data = {}
    for key in ['channel', 'phase']:
        pick_features = []
        if getattr(feature, key):
            for context_data in getattr(feature, key):
                pick_feature = _bytes_feature(context_data)
                pick_features.append(pick_feature)

        sequence_data[key] = tf.train.FeatureList(feature=pick_features)

    feature_lists = tf.train.FeatureLists(feature_list=sequence_data)

    example = tf.train.SequenceExample(context=context,
                                       feature_lists=feature_lists)
    return example.SerializeToString()


def sequence_example_parser(record):
    """
    Returns parsed example from sequence example.

    :param record: TFRecord.
    :return: Parsed example.
    """
    context = {
        "id": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "starttime": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "endtime": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "station": tf.io.FixedLenFeature((), tf.string, default_value=""),

        "npts": tf.io.FixedLenFeature(
            (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "delta": tf.io.FixedLenFeature(
            (), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),

        "trace": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "predict": tf.io.FixedLenFeature((), tf.string, default_value=""),
    }
    sequence = {
        "channel": tf.io.VarLenFeature(tf.string),
        "phase": tf.io.VarLenFeature(tf.string),
    }

    parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
        record,
        context_features=context,
        sequence_features=sequence)
    parsed_example = {
        'id': parsed_context['id'],
        'station': parsed_context['station'],
        'starttime': parsed_context['starttime'],
        'endtime': parsed_context['endtime'],

        'npts': parsed_context['npts'],
        'delta': parsed_context['delta'],

        "channel": tf.RaggedTensor.from_sparse(parsed_sequence['channel']),
        "phase": tf.RaggedTensor.from_sparse(parsed_sequence['phase']),
    }

    for trace in ['trace', 'label', 'predict']:
        trace_data = tf.io.decode_raw(parsed_context[trace], tf.float32)
        parsed_example[trace] = tf.reshape(
            trace_data, [1, parsed_example['npts'], -1])

    return parsed_example


def eval_eager_tensor(parsed_example):
    """
    Returns feature dict from parsed example.

    :param parsed_example: Parsed example.
    :return: Feature dict.
    """
    feature = Feature()

    feature.id = parsed_example['id'].numpy().decode('utf-8')
    feature.station = parsed_example['station'].numpy().decode('utf-8')
    feature.starttime = parsed_example['starttime'].numpy().decode('utf-8')
    feature.endtime = parsed_example['endtime'].numpy().decode('utf-8')

    feature.npts = parsed_example['npts'].numpy()
    feature.delta = parsed_example['delta'].numpy()

    feature.trace = parsed_example['trace']
    feature.channel = parsed_example['channel']

    feature.phase = parsed_example['phase']
    feature.label = parsed_example['label']
    feature.predict = parsed_example['predict']

    for key in ['channel', 'phase']:
        data_list = []
        for bytes_string in getattr(feature, key):
            item = bytes_string[0].numpy().decode('utf-8')
            data_list.append(item)
        setattr(feature, key, data_list)

    return feature


def batch_iterator(batch):
    """
    Yield parsed example from list.

    :param batch: list of parsed example.
    """
    for index in range(batch['id'].shape[0]):
        parsed_example = {
            'id': batch['id'][index],
            'station': batch['station'][index],
            'starttime': batch['starttime'][index],
            'endtime': batch['endtime'][index],

            'npts': batch['npts'][index],
            'delta': batch['delta'][index],

            "trace": batch['trace'][index, :],
            "channel": batch['channel'][index, :],

            "phase": batch['phase'][index, :],
            "label": batch['label'][index, :],
            "predict": batch['predict'][index, :],
        }
        yield parsed_example


if __name__ == "__main__":
    pass
