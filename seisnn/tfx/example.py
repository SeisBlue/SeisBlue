import tensorflow as tf
from obspy import read


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def trace_tfexample(file):
    trace = read(file)

    feature = {
        'starttime': _bytes_feature(trace.stats.starttime),
        'endtime': _bytes_feature(trace.stats.endtime),

        'delta': _float_feature(trace.stats.delta),
        'npts': _int64_feature(trace.stats.npts),

        'data': _float_feature(trace.data),
        'label': _float_feature(value),

        'network': _bytes_feature(trace.stats.network),
        'location': _bytes_feature(trace.stats.location),
        'station': _bytes_feature(trace.stats.station),
        'channel': _bytes_feature(trace.stats.channel),

        'latitude': _float_feature(value),
        'longitude': _float_feature(value),

        'pick_time': _float_feature(value),
        'pick_phase': _bytes_feature(value),
        'pick_type': _bytes_feature(value),

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
