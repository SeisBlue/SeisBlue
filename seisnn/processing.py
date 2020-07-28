"""
Processing
"""

import numpy as np
import scipy
import scipy.stats as ss
from scipy import signal
import obspy


def get_window(pick, trace_length=30):
    """
    Returns time window from pick.

    :type pick: obspy.Pick
    :param pick: Pick object.
    :type trace_length: int
    :param trace_length: (Optional.) Trace length.
    :rtype: dict
    :return: Time window.
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
    Returns probability density function from stream.

    :type stream: obspy.Stream
    :param stream: Stream object.
    :type sigma: float
    :param sigma: Width of normalize distribution.
    :rtype: obspy.Stream
    :return: Stream with pdf.
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
    Extract pick from probability density function.

    :type feature: seisnn.core.Feature
    :param feature: Feature object.
    :type phase: str
    :param phase: Phase name.
    :type pick_set: str
    :param pick_set: Pick set name.
    :type height: float
    :param height: Height threshold, from 0 to 1.
    :type distance: int
    :param distance: Distance threshold.
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


def stream_preprocessing(stream, database):
    """
    Return processed stream with pdf.

    :type stream: obspy.Stream
    :param stream: Stream object.
    :type database: str
    :param database: SQL database root.
    :rtype: obspy.Stream
    :return: Processed Stream.
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

    :type stream: obspy.Stream
    :param stream: Stream object.
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

    :type stream: obspy.Stream
    :param stream: Stream object.
    :type points: int
    :param points: Trace data points.
    :rtype: obspy.Stream
    :return: Trimmed stream.
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
