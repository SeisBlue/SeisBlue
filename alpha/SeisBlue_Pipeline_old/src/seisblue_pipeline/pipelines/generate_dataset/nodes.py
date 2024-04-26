"""
This is a boilerplate pipeline 'generate_dataset'
generated using Kedro 0.18.2
"""
from datetime import datetime
import logging
# from src.seisblue_pipeline.seisblue import core
from ...seisblue import core
from obspy import UTCDateTime, Stream
from obspy.clients.filesystem import sds
import numpy as np
import collections
import h5py
import operator
import scipy


def get_picks(database, tag):
    """
    Returns list of pick object from pick table.

    :param str database:
    :param str tag: Pick tag
    :rtype: list[core.Pick]
    :return: List of Pick objects
    """
    log = logging.getLogger(__name__)
    db = core.Client(database=database)
    picks = db.get_picks(tag=tag)
    picks = sorted(picks, key=lambda pick: [pick.station, pick.time])
    for pick in picks:
        inventory = db.get_inventory(station=pick.station,
                                     time=UTCDateTime(pick.time))[0]
        inventory.transfer_to_dataclass()
        pick.transfer_to_dataclass()
        pick.inventory = inventory
    log.debug(f"Get {len(picks)} picks from database ({database}).")
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


def get_waveform_time_windows(picks, trace_length):
    """
    Returns TimeWindow objects of waveform by pick time.

    :param list[core.Pick] picks: List of pick.
    :param int trace_length: Length of trace.
    :rtype: list[core.TimeWindow]
    :return: List of TimeWindow object.
    """
    time_windows = []
    time_window = _get_time_window(trace_length=trace_length,
                                   anchor_time=UTCDateTime(0),
                                   shift="random")
    for pick in picks:
        starttime = time_window.starttime
        endtime = time_window.endtime
        if starttime < pick.time < endtime:
            continue
        elif endtime < pick.time < endtime + 30:
            time_window = _get_time_window(trace_length=trace_length,
                                           anchor_time=starttime + 30)
        else:
            time_window = _get_time_window(trace_length=trace_length,
                                           anchor_time=pick.time,
                                           shift="random")
        time_window.station = pick.inventory.station
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
    log = logging.getLogger(__name__)
    streams = []
    for time_window in time_windows:
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
            streams.append(stream_dict)
        except Exception as e:
            log.error(
                f"station = {time_window.station}, start time = {time_window.starttime}, error = {e}")
    log.debug(f"Get {len(streams)} stream.")
    return streams


def _trim_trace(stream, points=3001):
    """
    Return trimmed stream in a given length.

    :param Stream stream: Obspy Stream object.
    :param int points: Trace data length.
    :rtype: Stream
    :return: Trimmed stream.
    """
    log = logging.getLogger(__name__)
    trace = stream[0]
    start_time = trace.stats.starttime
    if trace.data.size > 1:
        dt = (trace.stats.endtime - trace.stats.starttime) / (
                trace.data.size - 1)
        end_time = start_time + 3 + dt * (points - 1)
    elif trace.data.size == 1:
        end_time = start_time
    else:
        log.warning("No data points in trace")
        return
    stream.trim(
        start_time + 3, end_time, nearest_sample=True, pad=True, fill_value=0
    )
    return stream


def signal_preprocessing(streams):
    """
    Return signal processed streams.

    :param list[collections.defaultdict[Any, Stream]] streams:
        List of dictionary contains geophone type and streams.
    :rtype: list[Stream]
    :return: List of signal processed stream.
    """
    log = logging.getLogger(__name__)
    processed_streams = []
    for geophone_type_stream_dict in streams:
        for geophone_type, stream in geophone_type_stream_dict.items():
            stream.detrend("demean")
            stream.detrend("linear")
            stream.filter("bandpass", freqmin=1, freqmax=45)
            for trace in stream:
                trace.normalize()
            stream.resample(100)
            stream = _trim_trace(stream)
            processed_streams.append(stream)
    log.debug(f'Get {len(processed_streams)} signal preprocessed stream.')
    return processed_streams


def _if_pick_in_stream(stream, pick):
    return pick.inventory.station == stream[0].stats.station and stream[
        0].stats.starttime <= pick.time <= stream[0].stats.endtime


def get_instances(streams, database, phase, tag, shape, half_width):
    """
    Returns list of Instance objects.

    :param list[Stream] streams: List of obspy Stream objects.
    :param str database:
    :param list[core.Pick] picks: List of Pick objects.
    :param str phase: PSN for example.
    :param str tag:
    :param str shape: Label shape, see scipy.signal.windows.get_window().
    :param int half_width:
    :rtype: list[core.Instance]
    :return: List of Instance objects.
    """
    log = logging.getLogger(__name__)
    db = core.Client(database=database)
    instance_list = []
    phase = list(phase)
    for stream in streams:
        # inventory
        inventory = db.get_inventory(network=stream[0].stats.network,
                                     station=stream[0].stats.station,
                                     time=stream[0].stats.starttime)[0]
        inventory = inventory.transfer_to_dataclass()
        # time_window
        time_window = core.TimeWindow(starttime=stream[0].stats.starttime,
                                      endtime=stream[0].stats.endtime,
                                      npts=stream[0].stats.npts,
                                      sampling_rate=stream[0].stats.sampling_rate,
                                      delta=stream[0].stats.delta)
        # traces
        traces = []
        for tr in stream:
            channel = tr.stats.channel
            trace = core.Trace(inventory=inventory,
                               time_window=time_window,
                               data=tr.data,
                               channel=channel)
            traces.append(trace)
        traces.sort(key=operator.attrgetter('channel'))

        # label
        label = core.Label(inventory=inventory,
                           time_window=time_window,
                           phase=phase,
                           tag=tag,
                           data=None)
        label.generate_pick_uncertainty_label(database, shape, half_width)

        # trace_id
        date = time_window.starttime.isoformat()
        trace_id = '.'.join([date, inventory.network, inventory.station])

        # instance
        instance = core.Instance(inventory=inventory,
                                 time_window=time_window,
                                 traces=traces,
                                 labels=[label],
                                 trace_id=trace_id)
        instance_list.append(instance)
    log.debug(f"Get {len(instance_list)} instances.")
    return instance_list


def _write_hdf5_layer(group, dictionary):
    """
    Write HDF5 dynamically.

    :param h5py._hl.group.Group group: group
    :param dict[any, any] dictionary:
    """
    log = logging.getLogger(__name__)
    for key, value in dictionary.items():
        try:
            if value is None:
                continue
            elif isinstance(value, dict):
                sub_group = group.create_group(key)
                _write_hdf5_layer(sub_group, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, (datetime, UTCDateTime)):
                group.attrs.create(key, value.isoformat())
            else:
                group.attrs.create(key, value)
        except Exception as e:
            log.error(f"key={key}, value={value}, error={e}")


def write_hdf5(instances, dataset_path):
    """
    Add instances into HDF5 file.

    :param list[core.Instance] instances: List of instances.
    :param str dataset_path: hdf5 filepath
    """
    log = logging.getLogger(__name__)
    count = 0

    with h5py.File(dataset_path, "w") as f:
        for instance in instances:
            HDFdict = instance.dict()

            if HDFdict['trace_id'] not in f.keys():
                instance_group = f.create_group(HDFdict['trace_id'])
                Traces = HDFdict.pop('traces')
                Labels = HDFdict.pop('labels')

                _write_hdf5_layer(instance_group, HDFdict)

                traces_group = instance_group.create_group('traces')
                for Trace in Traces:
                    channel_group = traces_group.create_group(Trace['channel'])
                    _write_hdf5_layer(channel_group, Trace)

                labels_group = instance_group.create_group('labels')
                for Label in Labels:
                    label_group = labels_group.create_group(Label['tag'])
                    Picks = Label.pop('picks')
                    _write_hdf5_layer(label_group, Label)
                    pick_group = label_group.create_group('picks')
                    if len(Picks) == 2:
                        for Pick in Picks:
                            phase_group = pick_group.create_group(Pick['phase'])
                            _write_hdf5_layer(phase_group, Pick)
                count += 1
    log.debug(f'Write {dataset_path} with {count} instances')


def add_waveforms(instances, database, dataset_path):
    """
    Add waveforms with hdf5 information into database.

    :param list[core.Instance] instances: List of instances.
    :param str database:
    :param str dataset_path: hdf5 filepath
    """
    log = logging.getLogger(__name__)
    db = core.Client(database, echo=False, build=False)
    file_type = dataset_path.split('.')[-1]
    if file_type in ['h5', 'hdf5']:
        try:
            db.add_waveforms(instances, dataset_path, remove_duplicates=False)
        except FileNotFoundError:
            log.debug(f'{dataset_path} not found.')
