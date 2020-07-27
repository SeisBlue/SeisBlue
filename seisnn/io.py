"""
Input / Output
======================

.. autosummary::
    :toctree: io

    database_to_tfrecord
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

"""

import collections
import functools
import multiprocessing as mp
import os
import pickle
import shutil

from lxml import etree
from obspy import Stream
from obspy.clients.filesystem import sds
from obspy.core import inventory
from obspy.io.nordic.core import read_nordic
import tensorflow as tf

from seisnn import example_proto
from seisnn import processing
from seisnn import utils


def read_dataset(dataset_dir):
    file_list = utils.get_dir_list(dataset_dir)
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(example_proto.sequence_example_parser,
                          num_parallel_calls=mp.cpu_count())
    return dataset


def read_pkl(pkl):
    """Read python pickle file."""
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
        return obj


def write_pkl(obj, file):
    """Write python pickle file."""
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def write_tfrecord(example_list, save_file):
    """Write TFRecord file."""
    with tf.io.TFRecordWriter(save_file) as writer:
        for example in example_list:
            writer.write(example)


def read_event_list(sfile_dir):
    config = utils.get_config()
    sfile_dir = os.path.join(config['CATALOG_ROOT'], sfile_dir)

    sfile_list = utils.get_dir_list(sfile_dir)
    print(f'Reading events from {sfile_dir}')

    events = utils.parallel(par=get_event, file_list=sfile_list)
    print(f'Read {len(events)} events\n')
    return events


def get_event(filename, debug=False):
    import warnings
    with warnings.catch_warnings():
        if not debug:
            warnings.simplefilter("ignore")
        events = []
        for file in filename:
            try:
                catalog = read_nordic(file)
            except Exception as err:
                if debug:
                    print(err)
                continue
            for event in catalog.events:
                events.append(event)
        return events


def read_sds(window):
    """Read SDS database"""
    config = utils.get_config()
    station = window['station']
    starttime = window['starttime']
    endtime = window['endtime'] + 0.1

    client = sds.Client(sds_root=config['SDS_ROOT'])
    stream = client.get_waveforms(network="*",
                                  station=station,
                                  location="*",
                                  channel="*",
                                  starttime=starttime,
                                  endtime=endtime)
    stream.sort(keys=['channel'], reverse=True)

    stream_dict = collections.defaultdict(Stream)
    for trace in stream:
        geophone_type = trace.stats.channel[0:2]
        stream_dict[geophone_type].append(trace)

    return stream_dict


def database_to_tfrecord(database, output):
    import itertools
    import operator
    from seisnn.sql import Client, Pick
    config = utils.get_config()
    dataset_dir = os.path.join(config['TFRECORD_ROOT'], output)
    utils.make_dirs(dataset_dir)

    db = Client(database)
    query = db.get_picks().order_by(Pick.station)
    picks_groupby_station = [list(g) for k, g in itertools.groupby(
        query, operator.attrgetter('station'))]

    par = functools.partial(_write_picked_stream, database=database)
    for station_picks in picks_groupby_station:
        station = station_picks[0].station
        file_name = '{}.tfrecord'.format(station)

        example_list = utils.parallel(par, station_picks)
        save_file = os.path.join(dataset_dir, file_name)
        write_tfrecord(example_list, save_file)
        print(f'{file_name} done')


def _write_picked_stream(batch_picks, database):
    example_list = []
    for pick in batch_picks:
        if not pick.phase in ['P']:
            continue
        window = pick.get_window(pick)
        streams = read_sds(window)

        for _, stream in streams.items():
            stream.station = pick.station
            stream = processing.stream_preprocessing(stream, database)
            feature = example_proto.stream_to_feature(stream)
            example = example_proto.feature_to_example(feature)
            example_list.append(example)
    return example_list


def write_station_dataset(dataset_output_dir, sds_root,
                          nslc,
                          start_time, end_time,
                          trace_length=30, sample_rate=100,
                          remove_dir=False):
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    client = sds.Client(sds_root=sds_root)
    net, sta, loc, chan = nslc
    t = start_time
    counter = 0

    while t < end_time:
        stream = client.get_waveforms(net, sta, loc, chan, t,
                                      t + trace_length + 1)
        stream = processing.signal_preprocessing(stream)
        points = trace_length * sample_rate + 1

        for trace in stream:
            try:
                processing.trim_trace(trace, points)

            except IndexError as err:
                print(err)
                stream.remove(trace)
                continue

            finally:
                trace.picks = []
                time_stamp = trace.stats.starttime.isoformat()
                trace.write(
                    dataset_output_dir + '/' + time_stamp + trace.get_id() + ".pkl",
                    format="PICKLE")
                counter += 1

    t += trace_length


def read_hyp(hyp):
    """Read STATION0.HYP file."""
    config = utils.get_config()
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
                lat = inventory.util.Latitude(lat)

                lon = (int(lon[0:3]) + float(lon[3:-1]) / 60) * EW
                lon = inventory.util.Longitude(lon)

                location = {'latitude': lat,
                            'longitude': lon,
                            'elevation': elev}
                geom[sta] = location
    print(f'read {len(geom)} stations from {hyp}')
    return geom


def write_hyp_station(geom, save_file):
    """Write STATION0.HYP file."""
    config = utils.get_config()
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

        hyp.append(
            f' {sta: >5}{lat: >2d}{lat_min:>5.2f}{NS}{lon: >3d}{lon_min:>5.2f}{EW}{elev: >4d}\n')
    hyp.sort()

    output = os.path.join(config['GEOM_ROOT'], save_file)
    with open(output, 'w') as f:
        f.writelines(hyp)


def read_kml_placemark(kml):
    """Read Google Earth KML file."""
    config = utils.get_config()
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
