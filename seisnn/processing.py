"""
Processing
=============

.. autosummary::
    :toctree: pick

    get_pdf
    get_picks_from_dataset
    get_picks_from_pdf
    get_time_residual
    get_window
    signal_preprocessing
    stream_preprocessing
    trim_trace
    validate_picks_nearby

"""

import numpy as np
import scipy
import scipy.stats as ss
from scipy import signal
import obspy


def get_window(pick, trace_length=30):
    """

    :param pick:
    :param trace_length:
    :return:
    """
    scipy.random.seed()
    pick_time = obspy.UTCDateTime(pick.time)

    starttime = pick_time - trace_length + np.random.random_sample() * trace_length
    endtime = starttime + trace_length

    window = {
        'starttime': starttime,
        'endtime': endtime,
        'station': pick.station,
    }
    return window


def get_pdf(stream, sigma=0.1):
    """

    :param stream:
    :param sigma:
    :return:
    """
    trace = stream[0]
    start_time = trace.stats.starttime
    x_time = trace.times(reftime=start_time)

    stream.pdf = np.zeros([3008, 2])
    stream.phase = []

    for i, phase in enumerate(['P', 'S']):
        if stream.picks.get(phase):
            stream.phase.append(phase)
        else:
            stream.phase.append('')
            continue

        phase_pdf = np.zeros((len(x_time),))
        for pick in stream.picks[phase]:
            pick_time = obspy.UTCDateTime(pick.time) - start_time
            pick_pdf = ss.norm.pdf(x_time, pick_time, sigma)

            if pick_pdf.max():
                phase_pdf += pick_pdf / pick_pdf.max()

        stream.pdf[:, i] = phase_pdf
    return stream


def get_picks_from_pdf(feature, phase, pick_set, height=0.5, distance=100):
    """

    :param feature:
    :param phase:
    :param pick_set:
    :param height:
    :param distance:
    """
    i = feature.phase.index(phase)
    peaks, properties = signal.find_peaks(feature.pdf[-1, :, i],
                                          height=height,
                                          distance=distance)

    for p in peaks:
        if p:
            pick_time = obspy.UTCDateTime(
                feature.starttime) + p * feature.delta
            feature.pick_time.append(pick_time.isoformat())
            feature.pick_phase.append(feature.phase[i])
            feature.pick_set.append(pick_set)


def get_picks_from_dataset(dataset):
    """

    :param dataset:
    :return:
    """
    pick_list = []
    trace = obspy.read(dataset, headonly=True).traces[0]
    picks = trace.picks
    pick_list.extend(picks)
    return pick_list


def validate_picks_nearby(val_pick, pred_pick, delta=0.1):
    """

    :param val_pick:
    :param pred_pick:
    :param delta:
    :return:
    """
    upper_bound = obspy.UTCDateTime(pred_pick['pick_time']) + delta
    lower_bound = obspy.UTCDateTime(pred_pick['pick_time']) - delta
    if lower_bound < obspy.UTCDateTime(val_pick['pick_time']) < upper_bound:
        return True
    else:
        return False


def get_time_residual(val_pick, pred_pick):
    """

    :param val_pick:
    :param pred_pick:
    :return:
    """
    residual = obspy.UTCDateTime(val_pick['pick_time']) \
               - obspy.UTCDateTime(pred_pick['pick_time'])
    return residual


def stream_preprocessing(stream, database):
    """
    Return processed stream with pdf.

    :param stream:
    :param database:
    :return:
    """
    from seisnn import sql
    db = sql.Client(database)
    stream = signal_preprocessing(stream)

    starttime = stream.traces[0].stats.starttime.datetime
    endtime = stream.traces[0].stats.endtime.datetime
    station = stream.traces[0].stats.station

    stream.picks = {}
    for phase in ["P", "S"]:
        picks = db.get_picks(starttime=starttime, endtime=endtime,
                             station=station, phase=phase)
        if picks:
            stream.picks[phase] = picks

    stream = get_pdf(stream)
    return stream


def signal_preprocessing(stream):
    """
    Return a signal processed stream.

    :param stream:
    :return:
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

    :param stream:
    :param points:
    :return:
    """
    trace = stream[0]
    start_time = trace.stats.starttime
    dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
    end_time = start_time + dt * (points - 1)
    stream.trim(start_time, end_time, nearest_sample=True, pad=True,
                fill_value=0)
    return stream


if __name__ == "__main__":
    pass
