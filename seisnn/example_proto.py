"""
Example Protocol
"""

import numpy as np
import tensorflow as tf


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


def stream_to_feature(stream):
    """
    Turn Stream object into feature dictionary

    :param stream: Preprocessed stream object from
        seisnn.processing.stream_preprocessing
    :rtype: dict
    :return: feature dict

    """
    trace = stream[0]

    feature = {
        'id': trace.id,
        'starttime': trace.stats.starttime.isoformat(),
        'endtime': trace.stats.endtime.isoformat(),
        'station': trace.stats.station,
        'npts': trace.stats.npts,
        'delta': trace.stats.delta,
    }

    channel = []
    trace = np.zeros([3008, 1])
    for i, comp in enumerate(['Z']):
        try:
            st = stream.select(component=comp)
            trace[:, i] = st.traces[0].data
            channel.append(st.traces[0].stats.channel)
        except IndexError:
            pass

    feature['trace'] = trace
    feature['channel'] = channel

    feature['phase'] = stream.phase
    feature['pdf'] = np.asarray(stream.pdf)

    return feature


def feature_to_example(stream_feature):
    """

    :param stream_feature:
    :return:
    """
    for key in ['trace', 'pdf']:
        if isinstance(stream_feature[key], tf.Tensor):
            stream_feature[key] = stream_feature[key].numpy()

    context_data = {
        'id': _bytes_feature(stream_feature['id']),
        'starttime': _bytes_feature(stream_feature['starttime']),
        'endtime': _bytes_feature(stream_feature['endtime']),
        'station': _bytes_feature(stream_feature['station']),

        'npts': _int64_feature(stream_feature['npts']),
        'delta': _float_feature(stream_feature['delta']),

        'trace': _bytes_feature(
            stream_feature['trace'].astype(dtype=np.float32).tostring()),
        'pdf': _bytes_feature(
            stream_feature['pdf'].astype(dtype=np.float32).tostring()),
    }
    context = tf.train.Features(feature=context_data)

    sequence_data = {}
    for key in ['channel', 'phase']:
        pick_features = []
        if stream_feature[key]:
            for context_data in stream_feature[key]:
                pick_feature = _bytes_feature(context_data)
                pick_features.append(pick_feature)

        sequence_data[key] = tf.train.FeatureList(feature=pick_features)

    feature_list = tf.train.FeatureLists(feature_list=sequence_data)

    example = tf.train.SequenceExample(context=context,
                                       feature_lists=feature_list)
    return example.SerializeToString()


def sequence_example_parser(record):
    """

    :param record:
    :return:
    """
    context = {
        "id": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "starttime": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "endtime": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "station": tf.io.FixedLenFeature((), tf.string, default_value=""),

        "npts": tf.io.FixedLenFeature((), tf.int64, default_value=tf.zeros([],
                                                                           dtype=tf.int64)),
        "delta": tf.io.FixedLenFeature((), tf.float32,
                                       default_value=tf.zeros([],
                                                              dtype=tf.float32)),

        "trace": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "pdf": tf.io.FixedLenFeature((), tf.string, default_value=""),
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
        'starttime': parsed_context['starttime'],
        'endtime': parsed_context['endtime'],

        'delta': parsed_context['delta'],
        'npts': parsed_context['npts'],

        'station': parsed_context['station'],

        "channel": tf.RaggedTensor.from_sparse(parsed_sequence['channel']),
        "phase": tf.RaggedTensor.from_sparse(parsed_sequence['phase']),
    }

    for trace in ['trace', 'pdf']:
        trace_data = tf.io.decode_raw(parsed_context[trace], tf.float32)
        parsed_example[trace] = tf.reshape(trace_data,
                                           [1, parsed_example['npts'], -1])

    return parsed_example


def eval_eager_tensor(parsed_example):
    """

    :param parsed_example:
    :return:
    """
    feature = {
        'id': parsed_example['id'].numpy().decode('utf-8'),
        'starttime': parsed_example['starttime'].numpy().decode('utf-8'),
        'endtime': parsed_example['endtime'].numpy().decode('utf-8'),

        'delta': parsed_example['delta'].numpy(),
        'npts': parsed_example['npts'].numpy(),

        'station': parsed_example['station'].numpy(),

        "trace": parsed_example['trace'],
        "channel": parsed_example['channel'],
        "pdf": parsed_example['pdf'],
        "phase": parsed_example['phase'],
    }

    for i in ['channel', 'phase']:
        feature_list = feature[i]
        data_list = []
        for f in feature_list:
            f = f[0].numpy().decode('utf-8')
            data_list.append(f)
        feature[i] = data_list

    return feature


def batch_iterator(batch):
    """

    :param batch:
    """
    for index in range(batch['id'].shape[0]):
        parsed_example = {
            'id': batch['id'][index],
            'starttime': batch['starttime'][index],
            'endtime': batch['endtime'][index],

            'delta': batch['delta'][index],
            'npts': batch['npts'][index],

            'station': batch['station'][index],

            "trace": batch['trace'][index, :],
            "channel": batch['channel'][index, :],
            "pdf": batch['pdf'][index, :],
            "phase": batch['phase'][index, :],
        }
        yield parsed_example


if __name__ == "__main__":
    pass
