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


def trace_to_example(trace):
    starttime = trace.stats.starttime
    endtime = trace.stats.endtime

    delta = trace.stats.delta
    npts = trace.stats.npts

    data = trace.data

    try:
        pdf = trace.pdf
    except AttributeError:
        pdf = np.zeros((len(trace.data),))

    network = trace.stats.network
    location = trace.stats.location
    station = trace.stats.station
    channel = trace.stats.channel

    try:
        latitude = trace.stats.coordinates['latitude']
        longitude = trace.stats.coordinates['longitude']
    except AttributeError:
        latitude = 0.0
        longitude = 0.0

    pick_time = []
    pick_phase = []
    pick_type = []

    try:
        for pick in trace.picks:
            pick_time.append(pick.time)
            pick_phase.append(pick.phase)
            pick_type.append(pick.type)

    except AttributeError:
        pick_time = 0.0
        pick_phase = "NA"
        pick_type = "NA"
    pick_time = np.asarray(pick_time)
    pick_phase = np.asarray(pick_phase)
    pick_type = np.asarray(pick_type)

    feature = {
        'starttime': _bytes_feature(starttime.isoformat().encode('utf-8')),
        'endtime': _bytes_feature(endtime.isoformat().encode('utf-8')),

        'delta': _float_feature(delta),
        'npts': _int64_feature(npts),

        'data': _bytes_feature(data.astype(dtype=np.float32).tostring()),
        'pdf': _bytes_feature(pdf.astype(dtype=np.float32).tostring()),

        'network': _bytes_feature(network.encode('utf-8')),
        'location': _bytes_feature(location.encode('utf-8')),
        'station': _bytes_feature(station.encode('utf-8')),
        'channel': _bytes_feature(channel.encode('utf-8')),

        'latitude': _float_feature(latitude),
        'longitude': _float_feature(longitude),

        'pick_time': _bytes_feature(pick_time.astype(dtype=np.float32).tostring()),
        'pick_phase': _bytes_feature(pick_phase.astype(dtype=object).tostring()),
        'pick_type': _bytes_feature(pick_type.astype(dtype=object).tostring()),

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def extract_trace_example(example):
    feature = {
        'starttime': example.features.feature['starttime'].bytes_list.value[0].decode('utf-8'),
        'endtime': example.features.feature['endtime'].bytes_list.value[0].decode('utf-8'),

        'delta': example.features.feature['delta'].float_list.value[0],
        'npts': example.features.feature['npts'].int64_list.value[0],

        'network': example.features.feature['network'].bytes_list.value[0].decode('utf-8'),
        'location': example.features.feature['location'].bytes_list.value[0].decode('utf-8'),
        'station': example.features.feature['station'].bytes_list.value[0].decode('utf-8'),
        'channel': example.features.feature['channel'].bytes_list.value[0].decode('utf-8'),

        'latitude': example.features.feature['latitude'].float_list.value[0],
        'longitude': example.features.feature['longitude'].float_list.value[0],


    }

    for item in ['data', 'pdf', 'pick_time']:
        data = example.features.feature[item].bytes_list.value[0]
        data = np.fromstring(data, dtype=np.float32)
        feature[item] = data

    for item in ['pick_phase', 'pick_type']:
        data = example.features.feature[item].bytes_list.value[0]
        data = np.fromstring(data, dtype=object)
        feature[item] = data



    return feature

