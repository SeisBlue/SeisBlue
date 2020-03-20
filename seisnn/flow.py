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

from seisnn.pick import search_pick, get_pdf


def stream_preprocessing(stream, pick_list, pick_time_key, geom):
    """
    Main data preprocessing flow.

    :param stream:
    :param pick_list:
    :param pick_time_key:
    :param geom:
    :return:
    """
    stream = signal_preprocessing(stream)
    stream = get_exist_picks(stream, pick_list, pick_time_key)
    stream = get_pdf(stream)
    stream = get_stream_geom(stream, geom)
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


def get_exist_picks(stream, pick_list, pick_time_key):
    """
    Search picks in a given time window.

    :param stream:
    :param pick_list:
    :param pick_time_key:
    :return:
    """
    picks = search_pick(pick_list, pick_time_key, stream)
    stream.picks = picks
    return stream


def get_stream_geom(stream, geom):
    """
    Get station name from header.

    :param stream:
    :param geom:
    :return:
    """
    station = stream.traces[0].stats.station
    stream.location = geom[station]
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
