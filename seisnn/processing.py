"""
Processing
"""
import functools
import os

import numpy as np
import obspy
import scipy.signal
import scipy.stats

from seisnn.data import core, example_proto, io, sql
from seisnn import utils


def generate_training_data(pick_list, dataset, database, chunk_size):
    """
    Generate TFrecords from database.

    :param pick_list: List of picks from Pick SQL query.
    :param str dataset: Output directory name.
    :param str database: SQL database.
    :param int chunk_size: Number of data stores in TFRecord.
    """
    config = utils.get_config()
    dataset_dir = os.path.join(config['DATASET_ROOT'], dataset)
    utils.make_dirs(dataset_dir)

    par = functools.partial(get_example_list, database=database)

    total_batch = int(len(pick_list) / chunk_size)
    batch_picks = utils.batch(pick_list, size=chunk_size)
    for index, picks in enumerate(batch_picks):
        example_list = utils.parallel(par, picks)

        file_name = f'{index:0>5}.tfrecord'
        save_file = os.path.join(dataset_dir, file_name)
        io.write_tfrecord(example_list, save_file)
        print(f'output {file_name} / {total_batch}')


def get_time_window(anchor_time, station, trace_length=30, shift=0):
    """
    Returns time window from anchor time.

    :param anchor_time: Anchor of the time window.
    :param str station: Station name.
    :param int trace_length: (Optional.) Trace length, default is 30.
    :param float or str shift: (Optional.) Shift in sec,
        if 'random' will shift randomly within the trace length.
    :rtype: dict
    :return: Time window.
    """
    if shift == 'random':
        rng = np.random.default_rng()
        shift = rng.random() * trace_length

    starttime = obspy.UTCDateTime(anchor_time) - shift
    endtime = starttime + trace_length

    window = {
        'starttime': starttime,
        'endtime': endtime,
        'station': station,
    }
    return window


def get_example_list(batch_picks, database):
    """
    Returns example list form list of picks and SQL database.

    :param list batch_picks: List of picks.
    :param str database: SQL database root.
    :return:
    """

    example_list = []
    for pick in batch_picks:
        window = get_time_window(anchor_time=pick.time,
                                 station=pick.station,
                                 shift='random')

        streams = io.read_sds(window)
        for _, stream in streams.items():
            stream = signal_preprocessing(stream)

            instance = core.Instance().from_stream(stream)
            instance.phase = ['EQ', 'P', 'S']
            instance.get_label(database, shape='triang')
            instance.predict = np.zeros(instance.label.shape)

            feature = instance.to_feature()
            example = example_proto.feature_to_example(feature)
            example_list.append(example)

    return example_list


def get_label(instance, database, shape, half_width=10):
    """
    Add generated label to stream.

    :param instance: Data instance object.
    :param str database: SQL database.
    :param str shape: Label shape, see scipy.signal.windows.get_window().
    :param int half_width: Label half width in data point.
    :rtype: np.array
    :return: Label.
    """
    db = sql.Client(database)

    label = np.zeros([instance.npts, 3])
    for i, phase in enumerate(instance.phase):
        picks = db.get_picks(from_time=instance.starttime.datetime,
                             to_time=instance.endtime.datetime,
                             station=instance.station,
                             phase=phase).all()

        for pick in picks:
            pick_time = obspy.UTCDateTime(pick.time) - instance.starttime
            pick_time_index = int(pick_time / instance.delta)
            label[pick_time_index, i] = 1

    for i, phase in enumerate(instance.phase):
        if phase == 'EQ':
            # Make EQ window start by P and end by S.
            label[:, 0] = label[:, 1] - label[:, 2]
            label[:, 0] = np.cumsum(label[:, 0])
            if np.any(label[:, 0] < 0):
                label[:, 0] += 1
        else:
            wavelet = scipy.signal.windows.get_window(shape, 2 * half_width)
            label[:, i] = scipy.signal.convolve(label[:, i], wavelet[1:],
                                                mode='same')
    return label


def get_picks_from_predict(instance, tag, database,
                           height=0.5, distance=100):
    """
    Extract pick from predict.

    :param instance: Data instance.
    :param str tag: Output pick tag name.
    :param str database: SQL database name.
    :param float height: Height threshold, from 0 to 1, default is 0.5.
    :param int distance: Distance threshold in data point.
    """
    db = sql.Client(database)
    for i in instance.predict.shape[2]:
        peaks, _ = scipy.signal.find_peaks(instance.predict[-1, :, i],
                                           height=height,
                                           distance=distance)

        for peak in peaks:
            if peak:
                pick_time = obspy.UTCDateTime(
                    instance.starttime) + peak * instance.delta

                db.add_pick(time=pick_time.datetime,
                            station=instance.station,
                            phase=instance.phase[i],
                            tag=tag)


def get_picks_from_dataset(dataset):
    """
    Returns pick list from TFRecord Dataset.

    :param dataset: TFRecord Dataset.
    :rtype: list
    :return: List of picks.
    """
    pick_list = []
    trace = obspy.read(dataset, headonly=True).traces[0]
    picks = trace.picks
    pick_list.extend(picks)
    return pick_list


def validate_picks_nearby(val_pick, pred_pick, delta=0.1):
    """
    Finds nearby picks within a given range.

    :param val_pick: Validate pick.
    :param pred_pick: Predict pick.
    :param delta: Range.
    :rtype: bool
    :return: If any validate pick is nearby predict pick.
    """
    upper_bound = obspy.UTCDateTime(pred_pick['pick_time']) + delta
    lower_bound = obspy.UTCDateTime(pred_pick['pick_time']) - delta
    if lower_bound < obspy.UTCDateTime(val_pick['pick_time']) < upper_bound:
        return True
    else:
        return False


def get_time_residual(val_pick, pred_pick):
    """
    Returns time difference from validate pick to predict pick.

    :param val_pick: Validation pick.
    :param pred_pick: Predict pick.
    :rtype: float
    :return: Residual time.
    """
    residual = obspy.UTCDateTime(val_pick['pick_time']) \
               - obspy.UTCDateTime(pred_pick['pick_time'])
    return residual


def signal_preprocessing(stream):
    """
    Return a signal processed stream.

    :param obspy.Stream stream: Stream object.
    :rtype: obspy.Stream
    :return: Processed stream.
    """
    stream.detrend('demean')
    stream.detrend('linear')
    stream.normalize()
    stream.resample(100)
    stream = trim_trace(stream)
    return stream


def trim_trace(stream, points=3008):
    """
    Return trimmed stream in a given length.

    :param obspy.Stream stream: Stream object.
    :param int points: Trace data points.
    :rtype: obspy.Stream
    :return: Trimmed stream.
    """
    trace = stream[0]
    start_time = trace.stats.starttime
    dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
    end_time = start_time + dt * (points - 1)
    stream.trim(start_time,
                end_time,
                nearest_sample=True,
                pad=True,
                fill_value=0)
    return stream


if __name__ == "__main__":
    pass
