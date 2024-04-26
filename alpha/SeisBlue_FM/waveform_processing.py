# -*- coding: utf-8 -*-
from seisblue import io, tool, SQL, core

import itertools
import argparse
from obspy import UTCDateTime, Stream
from obspy.clients.filesystem import sds
import numpy as np
import collections
import operator
import scipy
import os
from itertools import chain
from tqdm import tqdm
import h5py


def _get_time_window(trace_length, anchor_time, shift=0):
    """
    Returns TimeWindow object from anchor time.
    :param int trace_length: Length of trace.
    :param anchor_time: Anchor of the time window.
    :param float or str shift: (Optional.) Shift in sec,
        if 'random' will shift randomly within the trace length.
    :rtype: core.TimeWindow
    :return: TimeWindow object.
    """
    if shift == "random":
        rng = np.random.default_rng()
        shift = rng.random() * (trace_length - 3)
        shift = shift + 3

    time_window = core.TimeWindow(
        starttime=UTCDateTime(anchor_time) - shift,
        endtime=UTCDateTime(anchor_time) - shift + trace_length
    )
    return time_window


def get_waveform_time_windows(pick, trace_length):
    """
    Returns TimeWindow objects of waveform by pick time.
    :param core.event : event.
    :param int trace_length: Length of trace.
    :rtype: list[core.TimeWindow]
    :return: List of TimeWindow object.
    """

    time_window = _get_time_window(trace_length=trace_length,
                                   anchor_time=UTCDateTime(0),
                                   shift="random")
    starttime = time_window.starttime
    endtime = time_window.endtime
    if starttime < pick.time < endtime:
        pass
    elif endtime < pick.time < endtime + 30:
        time_window = _get_time_window(trace_length=trace_length,
                                       anchor_time=starttime + 30)
    else:
        time_window = _get_time_window(trace_length=trace_length,
                                       anchor_time=pick.time,
                                       shift="random")
    time_window.station = pick.inventory.station
    return time_window


def get_waveforms(time_window, stations, waveforms_dir, trim=True, channel="*",
                  fmtstr=None):
    """
    Returns list of stream by reading SDS database.
    :param core.TimeWindow time_window:
    :param str waveforms_dir:
    :param bool trim:
    :param str channel:
    :param fmtstr:
    :rtype: list[collections.defaultdict[Any, Stream]]
    :return: List of dictionary contains geophone type and streams.
    """
    geophone_type_stream = []
    for station in tqdm(stations):
        stream_dict = {}
        try:
            starttime = time_window.starttime
            endtime = time_window.endtime + 0.1
            client = sds.Client(sds_root=waveforms_dir)
            if fmtstr:
                client.FMTSTR = fmtstr

            stream = client.get_waveforms(network="*",
                                          station=station,
                                          location="*",
                                          channel=channel,
                                          starttime=starttime,
                                          endtime=endtime)
            if stream:
                if trim:
                    stream.trim(starttime,
                                endtime,
                                pad=True,
                                fill_value=int(np.average(stream[0].data)))

            stream.sort(keys=["channel"], reverse=True)
            stream_dict = collections.defaultdict(Stream)
            for traces in stream:
                geophone_type = traces.stats.channel[0:2]
                stream_dict[geophone_type].append(traces)
        except Exception as e:
            print(f"station = {time_window.station}, start time = {time_window.starttime}, error = {e}")
        geophone_type_stream.append([stream for stream in stream_dict.values()])
    return geophone_type_stream


def _trim_trace(stream, points=3001):
    """
    Return trimmed stream in a given length.
    :param Stream stream: Obspy Stream object.
    :param int points: Trace data length.
    :rtype: Stream
    :return: Trimmed stream.
    """
    trace = stream[0]
    start_time = trace.stats.starttime
    if trace.data.size > 1:
        dt = (trace.stats.endtime - trace.stats.starttime) / (
                trace.data.size - 1)
        end_time = start_time + 3 + dt * (points - 1)
    elif trace.data.size == 1:
        end_time = start_time
    else:
        print("No data points in trace")
        return
    stream.trim(
        start_time + 3, end_time, nearest_sample=True, pad=True, fill_value=0
    )
    return stream


def signal_preprocessing(geophone_type_stream):
    """
    Return signal processed streams.
    :param list[collections.defaultdict[Any, Stream]] streams:
        List of dictionary contains geophone type and streams.
    :rtype: list[Stream]
    :return: List of signal processed stream.
    """
    processed_stream = []
    for stream in geophone_type_stream:
        stream.detrend("demean")
        stream.detrend("linear")
        stream.filter("bandpass", freqmin=1, freqmax=45)
        # for trace in stream:
        #     trace.normalize()
        # stream.resample(100)
        # stream = _trim_trace(stream)
        processed_stream.append(stream)
    return processed_stream


def get_traces(stream, timewindow=None):
    traces = []
    for tr in stream:
        channel = tr.stats.channel
        trace = core.Trace(inventory=core.Inventory(station=stream[0].stats.station,
                                                    network=stream[0].stats.network),
                           timewindow=timewindow,
                           data=tr.data,
                           channel=channel)
        traces.append(trace)
    traces.sort(key=operator.attrgetter('channel'))
    return traces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = io.read_yaml(args.data_config_filepath)
    c = config['process_event']

    print(f"Start processing waveform.")

    time_windows = tool.parallel(picks,
                                 func=get_waveform_time_windows,
                                 trace_length=c['trace_length'])
    print(f'Get {len(time_windows)} time windows.')

    stream = get_waveforms(time_windows, c['waveforms_dir'])
    print(f'Get {len(stream)} stream.')

    processed_waveforms = tool.parallel(stream, func=signal_preprocessing)
    processed_waveforms = list(chain.from_iterable(sublist for sublist in processed_waveforms if sublist))
    print(f'Get {len(processed_waveforms)} processed waveforms.')

