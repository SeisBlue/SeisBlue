# -*- coding: utf-8 -*-
import seisblue

import argparse
from obspy import UTCDateTime, Stream
from obspy.clients.filesystem import sds
import numpy as np
import collections
import scipy
import os
from tqdm import tqdm
import h5py
from datetime import datetime, timedelta
import time
import pathlib

def get_picks(database, **kwargs):
    client = seisblue.SQL.Client(database)
    results = client.get_picks(**kwargs)
    picks = [pick.to_dataclass() for pick in results]
    return picks


def _get_time_window(trace_length, anchor_time, shift=0):
    """
    Returns TimeWindow object from anchor time.
    :param int trace_length: Length of trace.
    :param anchor_time: Anchor of the time window.
    :param float or str shift: (Optional.) Shift in sec,
        if 'random' will shift randomly within the trace length.
    :rtype: seisblue.core.TimeWindow
    :return: TimeWindow object.
    """
    if shift == "random":
        rng = np.random.default_rng()
        shift = rng.random() * (trace_length - 3)
        shift = shift + 3

    time_window = seisblue.core.TimeWindow(
        starttime=UTCDateTime(anchor_time) - shift,
        endtime=UTCDateTime(anchor_time) - shift + trace_length
    )
    return time_window


def get_waveform_time_windows(pick, trace_length):
    """
    Returns TimeWindow objects of waveform by pick time.
    :param list[seisblue.core.Pick] picks: List of pick.
    :param int trace_length: Length of trace.
    :rtype: list[seisblue.core.TimeWindow]
    :return: List of TimeWindow object.
    """

    time_window = _get_time_window(trace_length=trace_length,
                                   anchor_time=UTCDateTime(0),
                                   shift="random")
    starttime = time_window.starttime
    endtime = time_window.endtime
    if starttime < pick.time < endtime:
        pass
    elif endtime < pick.time < endtime + 20:
        time_window = _get_time_window(trace_length=trace_length,
                                       anchor_time=starttime + 20)
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
    :param list[seisblue.core.TimeWindow] time_windows:
    :param str waveforms_dir:
    :param bool trim:
    :param str channel:
    :param fmtstr:
    :rtype: list[collections.defaultdict[Any, Stream]]
    :return: List of dictionary contains geophone type and streams.
    """
    geophone_type_stream = []
    for time_window in tqdm(time_windows):
        try:
            stream_dict = {}
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
            geophone_type_stream.append(
                [stream for stream in stream_dict.values()])
        except Exception as e:
            print(
                f"station = {time_window.station}, start time = {time_window.starttime}, error = {e}")
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
                stream.filter("bandpass", freqmin=1, freqmax=45)
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
    :param list[seisblue.core.Pick] picks: List of Pick objects.
    :param str phase: PSN for example.
    :param str shape:
    :param str half_width:
    :param str tag:
    :rtype: list[seisblue.core.Instance]
    :return: List of Instance objects.
    """
    instances = []
    for stream in tqdm(stream_list):
        timewindow = get_timewindow(stream)
        inventory = seisblue.core.Inventory(network=stream[0].stats.network,
                                   station=stream[0].stats.station)

        features = get_features(stream)
        label = generate_pick_uncertainty_label(database, phase, shape,
                                                half_width, tag,
                                                inventory, timewindow)
        for pick in label.picks:
            get_snr(pick, features, timewindow)
        if not label:
            continue
        if np.any((label.data < 0) | (label.data > 1)):
            continue
        if len(features.data) != 3:
            continue
        label.timewindow = None

        instance = seisblue.core.Instance(inventory=inventory,
                                 timewindow=timewindow,
                                 features=features,
                                 labels=[label])

        instances.append(instance)
    return instances


def get_snr(pick, features, timewindow, second=1):
    try:
        vector = np.linalg.norm(features.data, axis=0)
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

def get_timewindow(stream):
    return seisblue.core.TimeWindow(starttime=stream[0].stats.starttime.datetime,
                           endtime=stream[0].stats.endtime.datetime,
                           npts=stream[0].stats.npts,
                           samplingrate=stream[0].stats.sampling_rate,
                           delta=stream[0].stats.delta)


def get_features(stream):
    channels = []
    waveforms = []
    stream.sort()
    for tr in stream:
        channels.append(tr.stats.channel)
        waveforms.append(tr.data)
    waveforms = np.array(waveforms)
    features = seisblue.core.Stream(data=waveforms,
                           channel=','.join(channels))
    return features


def generate_pick_uncertainty_label(database, phases, shape, half_width, tag,
                                    inventory, timewindow):
    ph_index = {}
    data = np.zeros([timewindow.npts, len(phases)])
    instances_picks = []
    for i, phase in enumerate(phases):
        ph_index[phase] = i

        picks = get_picks(
            database=database,
            from_time=timewindow.starttime,
            to_time=timewindow.endtime,
            station=inventory.station,
            phase=phase,
            tag=tag)
        instances_picks.extend(picks)

        for pick in picks:
            pick_time = UTCDateTime(pick.time) - UTCDateTime(
                timewindow.starttime)
            pick_time_index = int(pick_time / timewindow.delta)

            data[pick_time_index, i] = 1

    if 'E' in phases:
        eq_time = (data[:, ph_index["P"]] - data[:, ph_index["S"]])
        eq_time = np.cumsum(eq_time)
        if np.any(eq_time < 0):
            eq_time += 1
        data[:, ph_index["E"]] = eq_time

    for i, phase in enumerate(phases):
        if not phase == "E":
            wavelet = scipy.signal.windows.get_window(shape,
                                                      2 * int(half_width))
            x = scipy.signal.convolve(data[:, i],
                                      wavelet[1:], mode="same")
            if not np.logical_and(x >= 0, x <= 1).all():
                x = (x-np.min(x))/(np.max(x)-np.min(x))
            data[:, i] = x

    if 'N' in phases:
        # Make Noise window by 1 - P - S
        data[:, ph_index["N"]] = 1
        data[:, ph_index["N"]] -= data[:, ph_index["P"]]
        data[:, ph_index["N"]] -= data[:, ph_index["S"]]

    label = seisblue.core.Label(phase=phases,
                       tag=tag,
                       data=data.T,
                       picks=instances_picks,
                       timewindow=timewindow)
    return label


def write_hdf5(instances, **kwargs):
    """
    Add instances into HDF5 file.
    :param List[seisblue.core.Instance] instances: List of instances.
    """
    for instance in instances:
        filepath = instance.datasetpath
        f = None
        try:
            f = h5py.File(filepath, 'a')
            HDFdict = seisblue.utils.to_dict(instance)
            id = '.'.join([instance.timewindow.starttime.isoformat(),
                           instance.inventory.network,
                           instance.inventory.station])
            if id not in f.keys():
                instance_group = f.create_group(id)
                seisblue.io.write_hdf5_layer(instance_group, HDFdict)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if f is not None:
                f.close()


def save_data(instances, dataset_dir, database):
    client = seisblue.SQL.Client(database)
    seisblue.utils.check_dir(dataset_dir)
    instancesSQL = []
    date = instances[0].timewindow.starttime.date().isoformat()
    dataset = f'{instances[0].inventory.network}_{date}.hdf5'
    datasetpath = os.path.join(dataset_dir, dataset)
    pathlib.Path(datasetpath).unlink(missing_ok=True)
    for instance in tqdm(instances):
        instance.dataset = dataset
        instance.datasetpath = datasetpath
        instanceSQL = seisblue.core.WaveformSQL(instance)
        instancesSQL.append(instanceSQL)

    write_hdf5(instances)
    print(f'Add {len(instances)} into {datasetpath}.')
    client.add_waveforms(instancesSQL)
    print(f'Add {len(instancesSQL)} into database.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = seisblue.io.read_yaml(args.data_config_filepath)
    c = config['process_waveform']
    dataset_dir = c['dataset_dir']
    database = c['database']
    start_time = time.time()

    print(f"Start processing waveform.")
    client = seisblue.SQL.Client(database)
    # client.clear_table("waveform")
    filter_params = c['pick_filter']
    current_time = filter_params['from_time']
    while current_time < filter_params['to_time']:
        new_filter = {
            'from_time': current_time,
            'to_time': (current_time + timedelta(days=1)),
            'tag': filter_params['tag'],
        }
        picks = get_picks(c['database'], **new_filter)
        print(f'Get {len(picks)} picks.')
        if len(picks) > 0:
            time_windows = seisblue.utils.parallel(picks,
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
            save_data(instances, dataset_dir, database)

            current_time += timedelta(days=1)
    end_time = time.time()

    inspector = seisblue.SQL.DatabaseInspector(c['database'])
    inspector.waveform_summery()

    time_log = f'Running time {int((end_time-start_time)/60)} minutes.'
    with open('log.txt', 'w') as f:
        f.write(time_log)


