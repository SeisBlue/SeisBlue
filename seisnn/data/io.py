"""
Input / Output
"""

import collections
import multiprocessing as mp
import os
import warnings

from lxml import etree
from obspy import Stream
from obspy.core import inventory
from obspy.clients.filesystem import sds
import obspy.io.nordic.core
import tensorflow as tf

from seisnn.data import example_proto
from seisnn import utils


def read_dataset(dataset):
    """
    Returns TFRecord Dataset from TFRecord directory.

    :param str dataset_dir: Directory contains TFRecords.
    :rtype: tf.data.Dataset
    :return: A Dataset.
    """
    config = utils.get_config()
    dataset_dir = os.path.join(config['DATASET_ROOT'], dataset)
    file_list = utils.get_dir_list(dataset_dir)
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(example_proto.sequence_example_parser,
                          num_parallel_calls=mp.cpu_count())
    return dataset


def write_tfrecord(example_list, save_file):
    """
    Writes TFRecord from example protocol.

    :param list example_list: List of example protocol.
    :param save_file: Output file path.
    """
    with tf.io.TFRecordWriter(save_file) as writer:
        for example in example_list:
            writer.write(example)


def read_event_list(sfile_dir):
    """
    Returns event list from sfile directory.

    :param str sfile_dir: Directory contains SEISAN sfile.
    :rtype: list
    :return: list of event.
    """
    config = utils.get_config()
    sfile_dir = os.path.join(config['CATALOG_ROOT'], sfile_dir)

    sfile_list = utils.get_dir_list(sfile_dir)
    print(f'Reading events from {sfile_dir}')

    events = utils.parallel(par=get_event, file_list=sfile_list)
    print(f'Read {len(events)} events\n')
    return events


def get_event(filename, debug=False):
    """
    Returns obspy.event list from sfile.

    :param str filename: Sfile file path.
    :param bool debug: If False, warning from reader will be ignore,
        default to False.
    :rtype: list
    :return: List of events.
    """
    with warnings.catch_warnings():
        if not debug:
            warnings.simplefilter("ignore")

        events = []
        for file in filename:
            try:
                catalog = obspy.io.nordic.core.read_nordic(file)
            except Exception as err:
                if debug:
                    print(err)
                continue

            for event in catalog.events:
                events.append(event)

        return events


def read_sds(window):
    """
    Read SDS database.

    :param dict window: Time window.
    :rtype: dict
    :return: Dict contains all traces within the time window.
    """
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


def read_hyp(hyp):
    """
    Returns geometry from STATION0.HYP file.

    :param str hyp: STATION0.HYP name without directory.
    :rtype: dict
    :return: Geometry dict.
    """
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

                lat_degree = int(lat[0:2])
                lat_minute = float(lat[2:-1]) / 60

                if lat_minute > 1: # high accuracy lat-lon
                    lat_minute /= 1000

                lat = (lat_degree + lat_minute) * NS
                lat = inventory.util.Latitude(lat)

                lon_degree = int(lat[0:3])
                lon_minute = float(lat[3:-1]) / 60

                if lon_minute > 1:  # high accuracy lat-lon
                    lon_minute /= 1000

                lon = (lon_degree + lat_minute) * EW
                lon = inventory.util.Longitude(lon)

                location = {'latitude': lat,
                            'longitude': lon,
                            'elevation': elev}
                geom[sta] = location
    print(f'read {len(geom)} stations from {hyp}')
    return geom


def write_hyp_station(geom, save_file):
    """
    Write STATION0.HYP file from geometry.

    :param dict geom: Geometry dict.
    :param str save_file: Name of .HYP file.
    """
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
    """
    Returns geometry from Google Earth KML file.

    :param str kml: KML file name without directory.
    :rtype: dict
    :return: Geometry dict.
    """
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
