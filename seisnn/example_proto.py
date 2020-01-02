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
        'starttime': trace.stats.starttime,
        'endtime': trace.stats.endtime,
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
    context_data = {
        'id': _bytes_feature(stream_feature['id'].encode('utf-8')),
        'starttime': _bytes_feature(stream_feature['starttime'].isoformat().encode('utf-8')),
        'endtime': _bytes_feature(stream_feature['endtime'].isoformat().encode('utf-8')),
        'station': _bytes_feature(stream_feature['station'].encode('utf-8')),
        'npts': _int64_feature(stream_feature['npts']),
        'delta': _float_feature(stream_feature['delta']),
        'latitude': _float_feature(stream_feature['latitude']),
        'longitude': _float_feature(stream_feature['longitude']),
        'elevation': _float_feature(stream_feature['elevation'])
    }
    context = tf.train.Features(feature=context_data)

    # dict to list
    for key in ['channel', 'phase']:
        data_list = []
        for k, v in stream_feature[key].items():
            data_list.append(stream_feature[key][k].astype(dtype=np.float32).tostring())
        stream_feature[key] = list(stream_feature[key].keys())
        stream_feature[f'{key}_data'] = data_list

    # dataframe to list
    picks = stream_feature['picks']
    stream_feature['pick_time'] = picks['pick_time'].tolist()
    stream_feature['pick_phase'] = picks['pick_phase'].tolist()
    stream_feature['pick_set'] = picks['pick_set'].tolist()

    sequence_data = {}
    for key in ['pick_time', 'pick_phase', 'pick_set', 'channel', 'channel_data', 'phase', 'phase_data']:
        pick_features = []
        if stream_feature[key]:
            for context_data in stream_feature[key]:
                if isinstance(context_data, UTCDateTime):
                    pick_feature = _bytes_feature(context_data.isoformat().encode('utf-8'))
                elif isinstance(context_data, bytes):
                    pick_feature = _bytes_feature(context_data)
                else:
                    pick_feature = _bytes_feature(context_data.encode('utf-8'))
                pick_features.append(pick_feature)

        sequence_data[key] = tf.train.FeatureList(feature=pick_features)

    feature_list = tf.train.FeatureLists(feature_list=sequence_data)

    example = tf.train.SequenceExample(context=context, feature_lists=feature_list)
    return example.SerializeToString()


def sequence_example_parser(record):
    context = {
        "id": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "starttime": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "endtime": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "station": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "npts": tf.io.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "delta": tf.io.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
        "latitude": tf.io.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
        "longitude": tf.io.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
        "elevation": tf.io.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
    }
    sequence = {
        "pick_time": tf.io.VarLenFeature(tf.string),
        "pick_phase": tf.io.VarLenFeature(tf.string),
        "pick_set": tf.io.VarLenFeature(tf.string),

        "channel": tf.io.VarLenFeature(tf.string),
        "channel_data": tf.io.VarLenFeature(tf.string),
        "phase": tf.io.VarLenFeature(tf.string),
        "phase_data": tf.io.VarLenFeature(tf.string),
    }
    parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(record,
                                                                          context_features=context,
                                                                          sequence_features=sequence)
    parsed_example = {
        'id': parsed_context['id'],
        'starttime': parsed_context['starttime'],
        'endtime': parsed_context['endtime'],

        'delta': parsed_context['delta'],
        'npts': parsed_context['npts'],

        'station': parsed_context['station'],
        'latitude': parsed_context['latitude'],
        'longitude': parsed_context['longitude'],
        'elevation': parsed_context['elevation'],

        "pick_time": tf.RaggedTensor.from_sparse(parsed_sequence['pick_time']),
        "pick_phase": tf.RaggedTensor.from_sparse(parsed_sequence['pick_phase']),
        "pick_set": tf.RaggedTensor.from_sparse(parsed_sequence['pick_set']),

        "channel": tf.RaggedTensor.from_sparse(parsed_sequence['channel']),
        "channel_data": tf.RaggedTensor.from_sparse(parsed_sequence['channel_data']),
        "phase": tf.RaggedTensor.from_sparse(parsed_sequence['phase']),
        "phase_data": tf.RaggedTensor.from_sparse(parsed_sequence['phase_data']),
    }

    return parsed_example


def extract_parsed_example(parsed_example):
    feature = {
        'id': parsed_example['id'].numpy().decode('utf-8'),
        'starttime': UTCDateTime(parsed_example['starttime'].numpy().decode('utf-8')),
        'endtime': UTCDateTime(parsed_example['endtime'].numpy().decode('utf-8')),

        'delta': parsed_example['delta'].numpy(),
        'npts': parsed_example['npts'].numpy(),

        'station': parsed_example['station'].numpy().decode('utf-8'),
        'latitude': parsed_example['latitude'].numpy(),
        'longitude': parsed_example['longitude'].numpy(),
        'elevation': parsed_example['elevation'].numpy(),

        "pick_time": parsed_example['pick_time'],
        "pick_phase": parsed_example['pick_phase'],
        "pick_set": parsed_example['pick_set'],

        "channel": parsed_example['channel'],
        "channel_data": parsed_example['channel_data'],
        "phase": parsed_example['phase'],
        "phase_data": parsed_example['phase_data'],
    }

    for i in ['pick_time', 'pick_phase', 'pick_set', 'channel', 'phase']:
        feature_list = feature[i]
        data_list = []
        for f in feature_list:
            f = f[0].numpy().decode('utf-8')
            if i == 'pick_time':
                f = UTCDateTime(f)
            data_list.append(f)
        feature[i] = data_list

    picks = {'pick_time': feature.pop('pick_time'),
             'pick_phase': feature.pop('pick_phase'),
             'pick_set': feature.pop('pick_set')}

    feature['picks'] = pd.DataFrame.from_dict(picks)

    for types in ['channel', 'phase']:
        type_dict = {}
        for i, v in enumerate(feature[types]):
            type_dict[v] = np.fromstring(feature[f'{types}_data'].values[i].numpy(), dtype=np.float32)
        feature[types] = type_dict

    return feature


def extract_batch(index, batch):
    parsed_example = {
        'id': batch['id'][index],
        'starttime': batch['starttime'][index],
        'endtime': batch['endtime'][index],

        'delta': batch['delta'][index],
        'npts': batch['npts'][index],

        'station': batch['station'][index],
        'latitude': batch['latitude'][index],
        'longitude': batch['longitude'][index],
        'elevation': batch['elevation'][index],

        "pick_time": batch['pick_time'][index, :],
        "pick_phase": batch['pick_phase'][index, :],
        "pick_set": batch['pick_set'][index, :],

        "channel": batch['channel'][index, :],
        "channel_data": batch['channel_data'][index, :],
        "phase": batch['phase'][index, :],
        "phase_data": batch['phase_data'][index, :],
    }
    return parsed_example
