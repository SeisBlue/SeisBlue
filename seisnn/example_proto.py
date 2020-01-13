import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, str):
        value = value.encode('utf-8')
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

    pick_time = []
    pick_phase = []
    pick_set = []
    if stream.picks:
        for _, picks in stream.picks.items():
            for pick in picks:
                pick_time.append(pick.time.isoformat())
                pick_phase.append(pick.phase_hint)
                pick_set.append(pickset)
    feature['pick_time'] = pick_time
    feature['pick_phase'] = pick_phase
    feature['pick_set'] = pick_set

    return feature


def feature_to_example(stream_feature):
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

        'latitude': _float_feature(stream_feature['latitude']),
        'longitude': _float_feature(stream_feature['longitude']),
        'elevation': _float_feature(stream_feature['elevation']),

        'trace': _bytes_feature(stream_feature['trace'].astype(dtype=np.float32).tostring()),
        'pdf': _bytes_feature(stream_feature['pdf'].astype(dtype=np.float32).tostring()),
    }
    context = tf.train.Features(feature=context_data)

    sequence_data = {}
    for key in ['pick_time', 'pick_phase', 'pick_set', 'channel', 'phase']:
        pick_features = []
        if stream_feature[key]:
            for context_data in stream_feature[key]:
                pick_feature = _bytes_feature(context_data)
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

        "trace": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "pdf": tf.io.FixedLenFeature((), tf.string, default_value=""),
    }
    sequence = {
        "pick_time": tf.io.VarLenFeature(tf.string),
        "pick_phase": tf.io.VarLenFeature(tf.string),
        "pick_set": tf.io.VarLenFeature(tf.string),

        "channel": tf.io.VarLenFeature(tf.string),
        "phase": tf.io.VarLenFeature(tf.string),
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
        "phase": tf.RaggedTensor.from_sparse(parsed_sequence['phase']),
    }

    for trace in ['trace', 'pdf']:
        trace_data = tf.io.decode_raw(parsed_context[trace], tf.float32)
        parsed_example[trace] = tf.reshape(trace_data, [1, parsed_example['npts'], -1])

    return parsed_example


def eval_eager_tensor(parsed_example):
    feature = {
        'id': parsed_example['id'].numpy().decode('utf-8'),
        'starttime': parsed_example['starttime'].numpy().decode('utf-8'),
        'endtime': parsed_example['endtime'].numpy().decode('utf-8'),

        'delta': parsed_example['delta'].numpy(),
        'npts': parsed_example['npts'].numpy(),

        'station': parsed_example['station'].numpy(),
        'latitude': parsed_example['latitude'].numpy(),
        'longitude': parsed_example['longitude'].numpy(),
        'elevation': parsed_example['elevation'].numpy(),

        "pick_time": parsed_example['pick_time'],
        "pick_phase": parsed_example['pick_phase'],
        "pick_set": parsed_example['pick_set'],

        "trace": parsed_example['trace'],
        "channel": parsed_example['channel'],
        "pdf": parsed_example['pdf'],
        "phase": parsed_example['phase'],
    }

    for i in ['pick_time', 'pick_phase', 'pick_set', 'channel', 'phase']:
        feature_list = feature[i]
        data_list = []
        for f in feature_list:
            f = f[0].numpy().decode('utf-8')
            data_list.append(f)
        feature[i] = data_list

    return feature

def batch_iterator(batch):
    for index in range(batch['id'].shape[0]):
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

            "trace": batch['trace'][index, :],
            "channel": batch['channel'][index, :],
            "pdf": batch['pdf'][index, :],
            "phase": batch['phase'][index, :],
        }
        yield parsed_example
