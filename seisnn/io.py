import os
import pickle
import shutil
from functools import partial
from multiprocessing import cpu_count
import tensorflow as tf

from obspy import Stream
from obspy.clients.filesystem.sds import Client
from obspy.io.nordic.core import read_nordic
from obspy.core.inventory.util import Latitude, Longitude

from seisnn.pick import get_window
from seisnn.flow import signal_preprocessing, stream_preprocessing, trim_trace
from seisnn.example_proto import stream_to_feature, feature_to_example, sequence_example_parser
from seisnn.utils import get_config, make_dirs, parallel, get_dir_list


def read_dataset(dataset_dir):
    file_list = get_dir_list(dataset_dir)
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(sequence_example_parser, num_parallel_calls=cpu_count())
    return dataset


def read_pkl(pkl):
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
        return obj


def write_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def write_tfrecord(example_list, save_file):
    with tf.io.TFRecordWriter(save_file) as writer:
        for example in example_list:
            writer.write(example)


def read_event_list(sfile_dir):
    sfile_list = get_dir_list(sfile_dir)
    print('reading events...')
    events = parallel(par=get_event, file_list=sfile_list, batch_size=1)
    return events


def get_event(filename):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            catalog, wavename = read_nordic(filename[0], return_wavnames=True)
            for event in catalog.events:
                for pick in event.picks:
                    pick.waveform_id.wavename = wavename
            return catalog.events
        except:
            pass


def read_sds(window):
    config = get_config()
    station = window['station']
    starttime = window['starttime']
    endtime = window['endtime']

    client = Client(sds_root=config['SDS_ROOT'])
    stream = client.get_waveforms(network="*", station=station, location="*", channel="*",
                                  starttime=starttime, endtime=endtime)

    stream.sort(keys=['channel'], reverse=True)
    stream_list = {}

    for trace in stream:
        geophone_type = trace.stats.channel[0:2]
        if not stream_list.get(geophone_type):
            stream_list[geophone_type] = Stream(trace)
        else:
            stream_list[geophone_type].append(trace)

    return stream_list


def write_training_dataset(pick_list, geom, dataset, pickset, batch_size=20):
    config = get_config()
    dataset_dir = os.path.join(config['DATASET_ROOT'], dataset)
    make_dirs(dataset_dir)

    pick_time_key = []
    for pick in pick_list:
        pick_time_key.append(pick.time)

    par = partial(_write_picked_stream,
                  pick_list=pick_list,
                  pick_time_key=pick_time_key,
                  geom=geom,
                  pickset=pickset)

    example_list = parallel(par, pick_list, batch_size)
    
    station = pick_list[0].waveform_id.station_code
    file_name = '{}.tfrecord'.format(station)
    save_file = os.path.join(dataset_dir, file_name)
    
    write_tfrecord(example_list, save_file)


def _write_picked_stream(batch_picks, pick_list, pick_time_key, geom, pickset):
    example_list = []
    for pick in batch_picks:
        window = get_window(pick)
        streams = read_sds(window)

        for _, stream in streams.items():
            stream = stream_preprocessing(stream, pick_list, pick_time_key, geom)
            feature = stream_to_feature(stream, pickset)
            example = feature_to_example(feature)
            example_list.append(example)
    return example_list


def write_station_dataset(dataset_output_dir, sds_root, nslc, start_time, end_time,
                          trace_length=30, sample_rate=100, remove_dir=False):
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    client = Client(sds_root=sds_root)
    net, sta, loc, chan = nslc
    t = start_time
    counter = 0

    while t < end_time:
        stream = client.get_waveforms(net, sta, loc, chan, t, t + trace_length + 1)
        stream = signal_preprocessing(stream)
        points = trace_length * sample_rate + 1

        for trace in stream:
            try:
                trim_trace(trace, points)

            except IndexError as err:
                print(err)
                stream.remove(trace)
                continue

            finally:
                trace.picks = []
                time_stamp = trace.stats.starttime.isoformat()
                trace.write(dataset_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")
                counter += 1

    t += trace_length


def read_geom(hyp):
    config = get_config()
    hyp = os.path.join(config['GEOM_ROOT'], hyp)
    geom = {}
    with open(hyp, 'r') as file:
        blank_line = 0
        while True:
            line = file.readline().rstrip()

            if not len(line):
                blank_line += 1
                continue

            if blank_line > 1:
                break

            elif blank_line == 1:
                lat = line[6:14]
                lon = line[14:23]
                elev = float(line[23:])
                sta = line[1:6].strip()

                if lat[-1] == 'S':
                    NS = -1
                else:
                    NS = 1

                if lon[-1] == 'W':
                    EW = -1
                else:
                    EW = 1

                lat = (int(lat[0:2]) + float(lat[2:-1]) / 60) * NS
                lat = Latitude(lat)

                lon = (int(lon[0:3]) + float(lon[3:-1]) / 60) * EW
                lon = Longitude(lon)

                location = {'latitude': lat,
                            'longitude': lon,
                            'elevation': elev}
                geom[sta] = location
    return geom
