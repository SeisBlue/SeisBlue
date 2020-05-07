"""
Data processing flow
====================

.. autosummary::
    :toctree: flow

    stream_preprocessing
    signal_preprocessing
    get_exist_picks
    get_stream_geom
    trim_trace

"""

from seisnn.pick import get_pdf
from seisnn.db import Client


def stream_preprocessing(stream, database):
    stream = signal_preprocessing(stream)
    stream = get_exist_picks(stream, database)
    stream = get_pdf(stream)
    return stream


def signal_preprocessing(stream):
    """
    Main signal processing flow.

    :param stream:
    :return:
    """
    stream.detrend('demean')
    stream.detrend('linear')
    stream.normalize()
    stream.resample(100)
    stream = trim_trace(stream)
    return stream


def get_exist_picks(stream, database):
    db = Client(database)
    starttime = stream.traces[0].stats.starttime.datetime
    endtime = stream.traces[0].stats.endtime.datetime
    station = stream.traces[0].stats.station
    phase_list = db.list_pick_phase()
    picks = {}
    for phase in phase_list:
        phase = phase[0]
        picks[phase] = db.get_picks(starttime=starttime, endtime=endtime,
                                    station=station, phase=phase)
    stream.picks = picks
    return stream


def trim_trace(stream, points=3008):
    """
    Trim traces in required length for unet.

    :param stream:
    :param points:
    :return:
    """
    trace = stream[0]
    start_time = trace.stats.starttime
    dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
    end_time = start_time + dt * (points - 1)
    stream.trim(start_time, end_time, nearest_sample=True, pad=True, fill_value=0)
    return stream

if __name__ == "__main__":
    pass
