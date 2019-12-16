import numpy as np
import pandas as pd
import tensorflow as tf

from obspy import UTCDateTime


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


def stream_to_feature(stream, pickset):
    trace = stream[0]

    feature = {
        'id': trace.id,
        'starttime': trace.stats.starttime.isoformat(),
        'endtime': trace.stats.endtime.isoformat(),
        'station': trace.stats.station,
        'npts': trace.stats.npts,
        'delta': trace.stats.delta,

        'latitude': stream.location['latitude'],
        'longitude': stream.location['longitude'],
        'elevation': stream.location['elevation'],
    }

    channel_dict = {}
    for trace in stream:
        channel_dict[trace.stats.channel] = trace.data
    feature['channel'] = channel_dict

    picks_dict = {'pick_time': [], 'pick_phase': [], 'pick_set': []}
    if stream.picks:
        for _, picks in stream.picks.items():
            for pick in picks:
                picks_dict['pick_time'].append(pick.time.isoformat())
                picks_dict['pick_phase'].append(pick.phase_hint)
                picks_dict['pick_set'].append(pickset)

    feature['picks'] = pd.DataFrame(picks_dict)
    feature['phase'] = stream.pdf

    return feature


def feature_to_example(stream_feature):
    data = {
        'id': _bytes_feature(stream_feature['id'].encode('utf-8')),
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
        for k, v in stream_feature[key].items():
            data[k] = _bytes_feature(stream_feature[key][k].astype(dtype=np.float32).tostring())
        stream_feature[key] = list(stream_feature[key].keys())  # replace dict by its own keys
    context = tf.train.Features(feature=data)

    picks = stream_feature['picks']
    stream_feature['pick_time'] = picks['pick_time'].tolist()
    stream_feature['pick_phase'] = picks['pick_phase'].tolist()
    stream_feature['pick_set'] = picks['pick_set'].tolist()

    data_dict = {}
    for key in ['pick_time', 'pick_phase', 'pick_set', 'channel', 'phase']:
        pick_features = []
        if stream_feature[key]:
            for data in stream_feature[key]:
                pick_feature = _bytes_feature(data.encode('utf-8'))
                pick_features.append(pick_feature)

        data_dict[key] = tf.train.FeatureList(feature=pick_features)

    feature_list = tf.train.FeatureLists(feature_list=data_dict)

    example = tf.train.SequenceExample(context=context, feature_lists=feature_list)
    return example.SerializeToString()


def extract_example(example):
    example = tf.train.SequenceExample.FromString(example.numpy())
    feature = {
        'id': example.context.feature['id'].bytes_list.value[0].decode('utf-8'),
        'starttime': UTCDateTime(example.context.feature['starttime'].bytes_list.value[0].decode('utf-8')),
        'endtime': UTCDateTime(example.context.feature['endtime'].bytes_list.value[0].decode('utf-8')),

        'delta': example.context.feature['delta'].float_list.value[0],
        'npts': example.context.feature['npts'].int64_list.value[0],

        'station': example.context.feature['station'].bytes_list.value[0].decode('utf-8'),
        'latitude': example.context.feature['latitude'].float_list.value[0],
        'longitude': example.context.feature['longitude'].float_list.value[0],
        'elevation': example.context.feature['elevation'].float_list.value[0],
    }
    # get list
    for i in ['pick_time', 'pick_phase', 'pick_set', 'channel', 'phase']:
        feature_list = example.feature_lists.feature_list[i].feature
        data_list = []
        for f in feature_list:
            f = f.bytes_list.value[0].decode('utf-8')
            if i == 'pick_time':
                f = UTCDateTime(f)
            data_list.append(f)
            feature[i] = data_list

    # get trace
    for types in ['channel', 'phase']:
        type_dict = {}
        for i in feature[types]:
            sequence_data = example.context.feature[i].bytes_list.value[0]
            sequence_data = np.fromstring(sequence_data, dtype=np.float32)
            type_dict[i] = sequence_data
        feature[types] = type_dict

    picks = {'pick_time': feature.pop('pick_time'),
             'pick_phase': feature.pop('pick_phase'),
             'pick_set': feature.pop('pick_set')}

    feature['picks'] = pd.DataFrame.from_dict(picks)

    return feature
