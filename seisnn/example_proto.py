import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def stream_to_feature(stream):
    trace = stream[0]

    feature = {
        'starttime': trace.stats.starttime.isoformat(),
        'endtime': trace.stats.endtime.isoformat(),
        'station': trace.stats.station,
        'npts': trace.stats.npts,
        'delta': trace.stats.delta,

        'latitude': stream.location['latitude'],
        'longitude': stream.location['longitude'],
        'elevation': stream.location['elevation'],
    }

    channel_list = []
    for trace in stream:
        channel = trace.stats.channel
        channel_list.append(channel)
        feature[channel] = trace.data
    feature['channel'] = channel_list

    phase_list = []
    pick_time = []
    pick_phase = []
    pick_type = []

    if stream.picks:
        for phase, picks in stream.picks.items():
            phase_list.append(phase)
            for pick in picks:
                pick_time.append(pick.time.isoformat())
                pick_phase.append(pick.phase_hint)
                pick_type.append(pick.evaluation_mode)

    feature['phase'] = phase_list
    feature['pick_time'] = pick_time
    feature['pick_phase'] = pick_phase
    feature['pick_type'] = pick_type

    for phase, pdf in stream.pdf.items():
        feature[phase] = pdf

    return feature


def feature_to_example(stream_feature):
    data = {
        'starttime': _bytes_feature(stream_feature['starttime'].encode('utf-8')),
        'endtime': _bytes_feature(stream_feature['endtime'].encode('utf-8')),
        'station': _bytes_feature(stream_feature['station'].encode('utf-8')),
        'npts': _int64_feature(stream_feature['npts']),
        'delta': _float_feature(stream_feature['delta']),
        'latitude': _float_feature(stream_feature['latitude']),
        'longitude': _float_feature(stream_feature['longitude']),
        'elevation': _float_feature(stream_feature['elevation'])
    }

    for key in ['channel', 'phase']:
        for k in stream_feature[key]:
            data[k] = _bytes_feature(stream_feature[k].astype(dtype=np.float32).tostring())
    context = tf.train.Features(feature=data)

    data_dict = {}
    for key in ['pick_time', 'pick_phase', 'pick_type']:
        pick_features = []
        if stream_feature[key]:
            for data in stream_feature[key]:
                pick_feature = _bytes_feature(data.encode('utf-8'))
                pick_features.append(pick_feature)
        else:
            pick_features.append(_bytes_feature('NA'.encode('utf-8')))

        data_dict[key] = tf.train.FeatureList(feature=pick_features)

    feature_list = tf.train.FeatureLists(feature_list=data_dict)

    example = tf.train.SequenceExample(context=context, feature_lists=feature_list)
    return example.SerializeToString()
