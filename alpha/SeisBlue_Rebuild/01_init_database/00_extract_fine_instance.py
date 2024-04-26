# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from seisblue import io, utils, SQL, core, plot
from typing import List

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
from datetime import datetime, timedelta


def get_picks(obspy_event: list, tag: str, network=None) -> List[object]:
    """
    Returns list of picks dataclass from events list.
    :param list obspy_events: Obspy Event.
    :param str tag: Pick tag.
    :rtype: list
    :return: Dataclass Pick.
    """
    picks = []
    for pick in obspy_event.picks:
        time = pick.time
        station = pick.waveform_id.station_code
        network = pick.waveform_id.network_code or network
        phase = pick.phase_hint
        polarity = None if phase == 'S' else (pick.polarity or 'undecidable')
        pick = core.Pick(time=datetime.utcfromtimestamp(time.timestamp),
                         inventory=core.Inventory(station=station, network=network),
                         phase=phase,
                         tag=tag,
                         polarity=polarity,
                         )
        picks.append(pick)
    return picks


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
    :param list[core.Pick] picks: List of pick.
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


def get_waveforms(time_windows, waveforms_dir, trim=True, channel="*",
                  fmtstr=None):
    """
    Returns list of stream by reading SDS database.
    :param list[core.TimeWindow] time_windows:
    :param str waveforms_dir:
    :param bool trim:
    :param str channel:
    :param fmtstr:
    :rtype: list[collections.defaultdict[Any, Stream]]
    :return: List of dictionary contains geophone type and streams.
    """
    geophone_type_stream = []
    for time_window in tqdm(time_windows):
        stream_dict = {}
        try:
            station = time_window.station
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


def _trim_trace(stream, points=2048):
    """
    Return trimmed stream in a given length.
    :param Stream stream: Obspy Stream object.
    :param int points: Trace data length.
    :rtype: Stream
    :return: Trimmed stream.
    """
    trace = stream[0]
    start_time = trace.stats.starttime
    shift_seconds = 3
    if trace.data.size > 1:
        dt = (trace.stats.endtime - trace.stats.starttime) / (
                trace.data.size - 1)
        end_time = start_time + shift_seconds + dt * (points - 1)
        stream.trim(
            start_time + shift_seconds, end_time, nearest_sample=True, pad=True,
            fill_value=0
        )
    elif trace.data.size == 1:
        print("Only one data points in trace")
        return
    else:
        print("No data points in trace")
        return

    return stream


def signal_preprocessing(geophone_type_stream_group):
    """
    Return signal processed streams.
    :param list[collections.defaultdict[Any, Stream]] streams:
        List of dictionary contains geophone type and streams.
    :rtype: list[Stream]
    :return: List of signal processed stream.
    """
    processed_stream = []
    for i, geophone_type_stream in enumerate(geophone_type_stream_group):
        for stream in geophone_type_stream:
            try:
                stream.detrend("demean")
                stream.detrend("linear")
                stream.filter("bandpass", freqmin=1, freqmax=10)
                for trace in stream:
                    trace.normalize()
                stream.resample(100)
                stream = _trim_trace(stream)
                processed_stream.append(stream)
            except Exception as e:
                print(e)
    return processed_stream


def get_instance(stream_list, database, phase, shape, half_width, tag):
    """
    Returns list of Instance objects.
    :param str database:
    :param list[Stream] streams: List of obspy Stream objects.
    :param list[core.Pick] picks: List of Pick objects.
    :param str phase: PSN for example.
    :param str shape:
    :param str half_width:
    :param str tag:
    :rtype: list[core.Instance]
    :return: List of Instance objects.
    """
    instances = []
    for stream in tqdm(stream_list):
        timewindow = get_timewindow(stream)
        inventory = core.Inventory(network=stream[0].stats.network,
                                   station=stream[0].stats.station)

        traces = get_traces(stream, timewindow)
        label = get_label(database, phase, timewindow, shape, half_width, tag, inventory)
        id = '.'.join([timewindow.starttime.isoformat(), stream[0].stats.network, stream[0].stats.station])
        instance = core.Instance(inventory=inventory,
                                 timewindow=timewindow,
                                 traces=traces,
                                 labels=[label],
                                 id=id)
        instances.append(instance)
    return instances


def get_timewindow(stream):
    return core.TimeWindow(starttime=stream[0].stats.starttime.datetime,
                           endtime=stream[0].stats.endtime.datetime,
                           npts=stream[0].stats.npts,
                           samplingrate=stream[0].stats.sampling_rate,
                           delta=stream[0].stats.delta)


def get_traces(stream, timewindow=None):
    traces = []
    for tr in stream:
        channel = tr.stats.channel
        trace = core.Trace(inventory=core.Inventory(station=tr.stats.station,
                                                    network=tr.stats.network),
                           timewindow=timewindow,
                           data=tr.data,
                           channel=channel)
        traces.append(trace)
    traces.sort(key=operator.attrgetter('channel'))
    return traces


def get_label(database, phase, timewindow, shape=None, half_width=None, tag=None, inventory=None):
    label = core.Label(inventory=inventory,
                       timewindow=timewindow,
                       phase=phase,
                       tag=tag,
                       data=None)
    label = generate_pick_uncertainty_label(database, label, phase, shape, half_width)
    return label


def get_SQL_picks(database, **kwargs):
    client = SQL.Client(database)
    results = client.get_picks(**kwargs)
    picks = [pick.to_dataclass() for pick in results]
    return picks


def generate_pick_uncertainty_label(database, label, phases, shape, half_width):
    ph_index = {}
    label.data = np.zeros([label.timewindow.npts, len(label.phase)])
    for i, phase in enumerate(phases):
        ph_index[phase] = i

        picks = get_SQL_picks(
            database=database,
            from_time=label.timewindow.starttime,
            to_time=label.timewindow.endtime,
            station=label.inventory.station,
            phase=phase,
            tag=label.tag)

        for pick in picks:
            pick_time = UTCDateTime(pick.time) - UTCDateTime(label.timewindow.starttime)
            pick_time_index = int(pick_time / label.timewindow.delta)

            label.data[pick_time_index, i] = 1
            label.picks.append(pick)
        picks_time = label.data.copy()
        wavelet = scipy.signal.windows.get_window(shape, 2 * int(half_width))
        label.data[:, i] = scipy.signal.convolve(label.data[:, i],
                                                 wavelet[1:], mode="same")

    if 'E' in label.phase:
        eq_time = (picks_time[:, ph_index["P"]] - picks_time[:, ph_index["S"]])
        eq_time = np.cumsum(eq_time)
        if np.any(eq_time < 0):
            eq_time += 1
        label.data[:, ph_index["E"]] = eq_time

    if 'N' in label.phase:
        # Make Noise window by 1 - P - S
        label.data[:, ph_index["N"]] = 1
        label.data[:, ph_index["N"]] -= label.data[:, ph_index["P"]]
        label.data[:, ph_index["N"]] -= label.data[:, ph_index["S"]]

    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = io.read_yaml(args.data_config_filepath)
    c = config['process_event']
    database = c['database']

    print("Processing events and picks.")
    obspy_events = io.get_obspy_events(c['events_dir'])
    picks = utils.parallel(obspy_events,
                           func=get_picks,
                           tag=c['tag'],
                           network=c['network'])
    fig_dir = './figure/waveform/'
    utils.check_dir(fig_dir)
    for i, picks_in_one_event in enumerate(picks):
        sfilename = obspy_events[i].comments[0].text
        time_windows = utils.parallel(picks_in_one_event,
                                      func=get_waveform_time_windows,
                                      trace_length=c['trace_length'])
        print(f'Get {len(time_windows)} time windows.')

        stream = get_waveforms(time_windows, c['waveforms_dir'])
        print(f'Get {len(stream)} stream.')

        processed_waveforms = signal_preprocessing(stream)
        print(f'Get {len(processed_waveforms)} processed waveforms.')
        instances = get_instance(processed_waveforms, database,
                                 **c['instance_parameters'])
        print(f'Get {len(instances)} instances.')

        for instance in instances:
            print(sfilename)
            plot.plot_dataset(instance, save_dir=fig_dir, title=sfilename)