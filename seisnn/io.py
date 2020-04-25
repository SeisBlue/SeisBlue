"""
Input / Output
======================

.. autosummary::
    :toctree: io

    get_event
    read_dataset
    read_event_list
    read_hyp
    read_kml_placemark
    read_pkl
    read_sds
    write_hyp_station
    write_pkl
    write_station_dataset
    write_tfrecord
    write_training_dataset

"""

import os
import pickle
import shutil
from lxml import etree
from functools import partial
from multiprocessing import cpu_count
import tensorflow as tf

from obspy import Stream
from obspy.clients.filesystem import sds
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


def read_event_list(sfile):
    config = get_config()
    sfile_dir = os.path.join(config['CATALOG_ROOT'], sfile)
    sfile_list = get_dir_list(sfile_dir)
    print(f'reading events from {sfile_dir}')
    events = parallel(par=get_event, file_list=sfile_list)
    print(f'read {len(events)} events from {sfile}')
    return events


def get_event(filename):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            events = []
            for file in filename:
                catalog, wavename = read_nordic(file, return_wavnames=True)
                for event in catalog.events:
                    for pick in event.picks:
                        pick.waveform_id.wavename = wavename
                    events.append(event)
            return events
        except:
            pass


def read_sds(window):
    config = get_config()
    station = window['station']
    starttime = window['starttime']
    endtime = window['endtime'] + 0.1

    client = sds.Client(sds_root=config['SDS_ROOT'])
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


def database_to_tfrecord(database, output):
    from seisnn.database import Client, Picks
    from itertools import groupby
    from operator import attrgetter
    config = get_config()
    dataset_dir = os.path.join(config['TFRECORD_ROOT'], output)
    make_dirs(dataset_dir)

    db = Client(database)
    query = db.get_picks().order_by(Picks.station)
    picks_groupby_station = [list(g) for k, g in groupby(query, attrgetter('station'))]

    par = partial(_write_picked_stream, database=database)
    for station_picks in picks_groupby_station:
        station = station_picks[0].station
        file_name = '{}.tfrecord'.format(station)

        example_list = parallel(par, station_picks)
        save_file = os.path.join(dataset_dir, file_name)
        write_tfrecord(example_list, save_file)
        print(f'{file_name} done')



def _write_picked_stream(batch_picks, database):
    example_list = []
    for pick in batch_picks:
        if not pick.phase in ['P']:
            continue
        window = get_window(pick)
        streams = read_sds(window)

        for _, stream in streams.items():
            stream.station = pick.station
            stream = stream_preprocessing(stream, database)
            feature = stream_to_feature(stream)
            example = feature_to_example(feature)
            example_list.append(example)
    return example_list


def write_station_dataset(dataset_output_dir, sds_root, nslc, start_time, end_time,
                          trace_length=30, sample_rate=100, remove_dir=False):
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    client = sds.Client(sds_root=sds_root)
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


def read_hyp(hyp):
    config = get_config()
    hyp_file = os.path.join(config['GEOM_ROOT'], hyp)
    geom = {}
    with open(hyp_file, 'r') as file:
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

                NS = 1
                if lat[-1] == 'S':
                    NS = -1

                EW = 1
                if lon[-1] == 'W':
                    EW = -1

                lat = (int(lat[0:2]) + float(lat[2:-1]) / 60) * NS
                lat = Latitude(lat)

                lon = (int(lon[0:3]) + float(lon[3:-1]) / 60) * EW
                lon = Longitude(lon)

                location = {'latitude': lat,
                            'longitude': lon,
                            'elevation': elev}
                geom[sta] = location
    print(f'read {len(geom)} stations from {hyp}')
    return geom


def write_hyp_station(geom, save_file):
    config = get_config()
    hyp = []
    for sta, loc in geom.items():
        lat = int(loc['latitude'])
        lat_min = (loc['latitude'] - lat) * 60

        NS = 'N'
        if lat < 0:
            NS = 'S'

        lon = int(loc['longitude'])
        lon_min = (loc['longitude'] - lon) * 60

        EW = 'E'
        if lat < 0:
            EW = 'W'

        elev = int(loc['elevation'])

        hyp.append(f' {sta: >5}{lat: >2d}{lat_min:>5.2f}{NS}{lon: >3d}{lon_min:>5.2f}{EW}{elev: >4d}\n')
    hyp.sort()

    output = os.path.join(config['GEOM_ROOT'], save_file)
    with open(output, 'w') as f:
        f.writelines(hyp)


def read_kml_placemark(kml):
    config = get_config()
    kml_file = os.path.join(config['GEOM_ROOT'], kml)

    parser = etree.XMLParser()
    root = etree.parse(kml_file, parser).getroot()
    geom = {}
    for Placemark in root.findall('.//Placemark', root.nsmap):
        sta = Placemark.find('.//name', root.nsmap).text
        coord = Placemark.find('.//coordinates', root.nsmap).text
        coord = coord.split(",")
        location = {'latitude': float(coord[1]),
                    'longitude': float(coord[0]),
                    'elevation': float(coord[2])}
        geom[sta] = location

    print(f'read {len(geom)} stations from {kml}')
    return geom

if __name__ == "__main__":
    pass
