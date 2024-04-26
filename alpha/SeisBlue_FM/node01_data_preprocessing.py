# -*- coding: utf-8 -*-
from typing import List
import glob
import os
from datetime import datetime
import itertools
from obspy import UTCDateTime, Stream
from obspy.clients.filesystem import sds
import numpy as np
import collections
import operator
from math import floor

from seisblue import core, tool, io
import inventory_processing as inv


def get_event(obspy_event):
    """
    Returns event objects from obspy events .
    :param list[obspy.core.event.event.Event] obspy_events: List of obspy events.
    :rtype: list[core.Event]
    :return: List of event objects.
    """
    origin_time = obspy_event.origins[0].time
    latitude = obspy_event.origins[0].latitude
    longitude = obspy_event.origins[0].longitude
    depth = obspy_event.origins[0].depth
    # magnitude = obspy_event.magnitudes[0].mag
    if len(obspy_event.focal_mechanisms) > 0:
        np1 = obspy_event.focal_mechanisms[0].nodal_planes.nodal_plane_1
        np = core.NodalPlane(strike=np1.strike,
                             strike_errors=np1.strike_errors.uncertainty,
                             dip=np1.dip,
                             dip_errors=np1.dip_errors.uncertainty,
                             rake=np1.rake,
                             rake_errors=np1.rake_errors.uncertainty)
    else:
        np = None

    event = core.Event(
        time=datetime.utcfromtimestamp(origin_time.timestamp),
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        focal_mechanism=np)

    return event


def get_picks(obspy_event, tag: str, geom, reverse=False, with_manual=False) -> List[object]:
    """
    Returns list of picks dataclass from events list.
    :param list obspy_events: Obspy Event.
    :param str tag: Pick tag.
    :rtype: list
    :return: Dataclass Pick.
    """
    picks = []
    reverse_dict = {'positive': 'negative', 'negative': 'positive',
                    'undecidable': 'undecidable'}
    for pick in obspy_event.picks:
        phase = pick.phase_hint
        try:
            arrival = [arr for arr in obspy_event.origins[0].arrivals if
                       arr.pick_id == pick.resource_id][0]
            time = pick.time
            station = pick.waveform_id.station_code
            inventory = [inv for inv in geom if inv.station == station][0]

            if with_manual:
                if phase == 'S':
                    polarity = None
                else:
                    polarity = pick.polarity if pick.polarity else 'undecidable'
                    if reverse:
                        polarity = reverse_dict[polarity]
            else:
                polarity = None

            pick = core.Pick(time=datetime.utcfromtimestamp(time.timestamp),
                             inventory=inventory,
                             phase=phase,
                             polarity=polarity,
                             tag=tag,
                             azimuth=arrival.azimuth,
                             takeoff_angle=arrival.takeoff_angle,
                             pick_id=arrival.resource_id.id)
            picks.append(pick)
        except Exception as e:
            print(
                f'{obspy_event.origins[0].time} {pick.waveform_id.station_code} fail to get picks.')

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


def get_waveform_time_windows(picks, trace_length, shift):
    """
    Returns TimeWindow objects of waveform by pick time.
    :param list[core.Pick] picks: List of pick.
    :param int trace_length: Length of trace.
    :rtype: list[core.TimeWindow]
    :return: List of TimeWindow object.
    """
    time_windows = []
    for pick in picks:
        if pick.phase != 'S':
            s_picks = [p for p in picks if
                       p.inventory.station == pick.inventory.station and p.phase == 'S']
            pick_pairs = [pick, s_picks[0]] if len(s_picks) > 0 else [pick]
            time_window = _get_time_window(trace_length=trace_length,
                                           anchor_time=UTCDateTime(pick.time),
                                           shift=shift)
            time_window.inventory = pick.inventory
            time_window.picks = pick_pairs
            time_windows.append(time_window)
    return time_windows


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
    streams = []
    for time_window in time_windows:
        starttime = time_window.starttime
        endtime = time_window.endtime + 0.1
        client = sds.Client(sds_root=waveforms_dir)
        if fmtstr:
            client.FMTSTR = fmtstr

        stream = client.get_waveforms(network="*",
                                      station=time_window.inventory.station,
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
        streams.append([stream_dict, time_window.picks])
    return streams


def _trim_trace(stream, points=256):
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
        dt = trace.stats.delta
        end_time = start_time + dt * (points - 1)
    elif trace.data.size == 1:
        end_time = start_time
    else:
        print("No data points in trace")
        return
    stream.trim(
        start_time, end_time, nearest_sample=True, pad=True, fill_value=0
    )
    return stream


def signal_preprocessing(streams, trace_length_npts):
    """
    Return signal processed streams.
    :param list[collections.defaultdict[Any, Stream]] streams:
        List of dictionary contains geophone type and streams.
    :rtype: list[Stream]
    :return: List of signal processed stream.
    """
    processed_streams = []
    for [geophone_type_stream_dict, picks] in streams:
        for geophone_type, stream in geophone_type_stream_dict.items():
            stream.detrend("demean")
            stream.detrend("linear")
            # down sampling prefilter to avoid aliasing(freqmax < resmaple/2)
            stream.filter("bandpass", freqmin=1, freqmax=45)

            for trace in stream:
                trace.normalize()
            stream.resample(100)
            stream = _trim_trace(stream, points=trace_length_npts)
            processed_streams.append((stream, picks))
    return processed_streams


def get_instances(waveforms_events, tag, dataset_name):
    stream, event = waveforms_events
    sub_instances = get_sub_instances(stream, tag)
    evt_id = event.time.isoformat()
    instance = core.EventInstance(
        instances=sub_instances,
        id=evt_id,
        event=event,
        dataset=dataset_name,
    )
    return instance


def get_sub_instances(stream, tag):
    sub_instaces = []
    for (stream, picks) in stream:
        timewindow = core.TimeWindow(
            starttime=stream[0].stats.starttime.datetime,
            endtime=stream[0].stats.endtime.datetime,
            npts=stream[0].stats.npts,
            samplingrate=stream[0].stats.sampling_rate,
            delta=stream[0].stats.delta)

        inventory = core.Inventory(
            network=stream[0].stats.network,
            station=stream[0].stats.station)
        traces = get_traces(stream, timewindow)

        for pick in picks:
            get_snr(pick, traces, timewindow)
        p_pick, s_pick = picks if len(picks) > 1 else [picks[0], None]

        id = '.'.join(
            [p_pick.time.isoformat(), stream[0].stats.network,
             stream[0].stats.station])

        label = get_label(p_pick, tag)
        instance = core.Instance(inventory=inventory,
                                 timewindow=timewindow,
                                 traces=traces,
                                 labels=[label],
                                 id=id,
                                 Spick=s_pick)
        sub_instaces.append(instance)
    return sub_instaces


def get_traces(stream, timewindow=None):
    traces = []
    for tr in stream:
        trace = core.Trace(inventory=core.Inventory(station=tr.stats.station,
                                                    network=tr.stats.network),
                           timewindow=timewindow,
                           data=tr.data,
                           channel=tr.stats.channel)
        traces.append(trace)
    traces.sort(key=operator.attrgetter('channel'))
    return traces


def get_label(pick, tag: str):
    label = core.Label(pick=pick,
                       tag=tag,
                       data=None)
    polarity = label.pick.polarity
    if polarity == 'positive':
        label.data = np.array([1, 0, 0])
    elif polarity == 'negative':
        label.data = np.array([0, 1, 0])
    else:
        label.data = np.array([0, 0, 1])
    return label


def get_snr(pick, traces, timewindow, second=1):
    try:
        data = np.array([trace.data for trace in traces])
        vector = np.linalg.norm(data, axis=0)
        point = int((pick.time - timewindow.starttime).total_seconds() * 100)
        if point >= second * 100:
            signal = vector[point: point + second * 100]
            noise = vector[point - len(signal): point]
        else:
            noise = vector[0:point]
            signal = vector[point: point + len(noise)]
        snr = signal_to_noise_ratio(signal=signal, noise=noise)
        pick.snr = np.around(snr, 4)
    except Exception as e:
        print(e)


def signal_to_noise_ratio(signal, noise):
    """
    Calculates power ratio from signal and noise.

    :param numpy.array signal: Signal trace data.
    :param numpy.array noise: Noise trace data.
    :rtype: float
    :return: Signal to noise ratio.
    """
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(noise))
    snr = np.log10(signal_power / noise_power)
    return snr


def save_data(instances, split_ratio={'test': 1}):
    dataset_dir = os.path.join('./dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    s_index = 0
    for mode, ratio in split_ratio.items():
        e_index = floor(len(instances) * ratio) + s_index
        instances_par = instances[s_index:e_index]
        instances_par = itertools.groupby(instances_par, key=lambda x: x.id[:7])
        io.write_hdf5(instances_par, mode, dataset_dir)
        s_index = e_index

def analysis_snr(eventinstances):
    p_snr = []
    s_snr = []
    for eventinstance in eventinstances:
        for instance in eventinstance.instances:
            if instance.labels[0].pick:
                p_snr.append(instance.labels[0].pick.snr)
            if instance.Spick:
                s_snr.append(instance.Spick.snr)
    print(p_snr[0])
    print(s_snr)
    seisblue.plot.

if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['process_event']

    print("Processing inventory.")
    station_time_dict = inv.get_stations_time_window(c['sub_waveforms_dir'])
    print(
        f'Get {len(station_time_dict)} stations with time window from {c["sub_waveforms_dir"]}')
    geom = inv.read_hyp(c['hyp_filepath'], station_time_dict)
    print(f'Read {len(geom)} stations in {c["hyp_filepath"]}')

    print("Processing events and picks.")
    obspy_events = io.get_obspy_events(c['events_dir'])[:10]
    print(f"Read {len(obspy_events)} events from {c['events_dir']}.")
    events = tool.parallel(obspy_events,
                           func=get_event,
                           order=True)
    picks = tool.parallel(obspy_events,
                          func=get_picks,
                          order=True,
                          tag=c['pick_tag'],
                          geom=geom,
                          reverse=c['reverse'],
                          with_manual=c['with_manual'])
    pick_PSs = [pick_PS for picks_PSs in picks for pick_PS in picks_PSs]
    p_number = len([pick for pick in pick_PSs if pick.phase == 'P'])
    s_number = len([pick for pick in pick_PSs if pick.phase == 'S'])
    print(f'Read {p_number} P and {s_number} S picks.')

    time_windows = tool.parallel(picks,
                                 func=get_waveform_time_windows,
                                 order=True,
                                 trace_length=c['trace_length'],
                                 shift=c['shift'])

    print(f'Start to get Stream.')
    waveforms = tool.parallel(time_windows,
                              func=get_waveforms,
                              waveforms_dir=c['waveforms_dir'],
                              trim=False,
                              order=True)


    print(f'Signal_preprocessing.')
    processed_waveforms = tool.parallel(waveforms,
                                        func=signal_preprocessing,
                                        order=True,
                                        trace_length_npts=c[
                                            'trace_length_npts'])
    instances = tool.parallel(list(zip(processed_waveforms, events)),
                              func=get_instances,
                              tag=c['label_tag'],
                              dataset_name=c['dataset_name'])
    print(f"Get instances.")

    analysis_snr(instances)
    # save_data(instances, c['split_ratio'])
    # print(f"Add instances.")
