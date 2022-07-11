"""
Input / Output
"""

import collections
import multiprocessing as mp
import itertools
import io
import os
import warnings

import numpy as np
import pandas as pd
from lxml import etree
from obspy import Stream, UTCDateTime
from obspy.core import inventory
from obspy.clients.filesystem import sds
import obspy.io.nordic.core
import obspy
import tensorflow as tf
from obspy.core.event import Event, EventDescription, Origin, WaveformStreamID, \
    Pick
from obspy.io.nordic.core import _write_nordic
from tensorflow.python.keras.utils.generic_utils import Progbar

import seisblue
import seisblue.example_proto
import seisblue.utils


def read_dataset(file_list):
    """
    Returns TFRecord Dataset from TFRecord directory.

    :param file_list: List of .tfrecord.
    :rtype: tf.data.Dataset
    :return: A Dataset.
    """

    dataset = tf.data.TFRecordDataset(file_list)
    dataset = parse_dataset(dataset)
    return dataset


def parse_dataset(dataset):
    dataset = dataset.map(seisblue.example_proto.sequence_example_parser,
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
    config = seisblue.utils.Config()
    sfile_dir = os.path.join(config.catalog, sfile_dir)

    sfile_list = seisblue.utils.get_dir_list(sfile_dir, ".S*")
    print(f'Reading events from {sfile_dir}')

    event_list = seisblue.utils.parallel(sfile_list, func=get_event)
    flatten = itertools.chain.from_iterable

    events = event_list
    while isinstance(events[0], list):
        events = flatten(events)
        events = [event for event in events if event]

    print(f'Read {len(events)} events\n')
    return events


def get_event(file, debug=False):
    """
    Returns obspy.event list from sfile.

    :param str file: Sfile file path.
    :param bool debug: If False, warning from reader will be ignore,
        default to False.
    :rtype: list
    :return: List of events.
    """
    with warnings.catch_warnings():
        if not debug:
            warnings.simplefilter("ignore")
        try:
            catalog = obspy.io.nordic.core.read_nordic(file)
            return catalog.events

        except Exception as err:
            if debug:
                print(err)


def read_sds(metadata, trim=True, channel='*', fmtstr=None):
    """
    Read SDS database.

    :param metadata: Metadata.
    :param trim:
    :param channel:
    :param fmtstr:
    :rtype: dict
    :return: Dict contains all traces within the time window.
    """
    config = seisblue.utils.Config()
    station = metadata.station
    starttime = metadata.starttime
    endtime = metadata.endtime + 0.1

    client = sds.Client(sds_root=config.sds_root)
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
            stream.trim(starttime, endtime, pad=True,
                        fill_value=int(np.average(stream[0].data)))
    stream.sort(keys=['channel'], reverse=True)

    stream_dict = collections.defaultdict(Stream)
    for trace in stream:
        geophone_type = trace.stats.channel[0:2]
        stream_dict[geophone_type].append(trace)
    return stream_dict


def read_gdms(metadata, trim=True, channel='*'):
    """
    Read CWB GDMS database.

    :param metadata: Metadata.
    :param trim:
    :param channel:
    :rtype: dict
    :return: Dict contains all traces within the time window.
    """

    fmtstr = os.path.join(
        "{year}", "{doy:03d}", "{station}",
        "{station}.{network}.{location}.{channel}.{year}.{doy:03d}")

    stream_dict = read_sds(metadata, trim, channel, fmtstr)
    return stream_dict


def read_hyp(hyp):
    """
    Returns geometry from STATION0.HYP file.

    :param str hyp: STATION0.HYP name without directory.
    :rtype: dict
    :return: Geometry dict.
    """
    config = seisblue.utils.Config()
    hyp_file = os.path.join(config.geom, hyp)
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
                lat, lon = convert_lat_lon(lat, lon)
                location = {'latitude': lat,
                            'longitude': lon,
                            'elevation': elev}
                geom[sta] = {'location': location}
    print(f'read {len(geom)} stations from {hyp}')
    return geom


def read_GDMSstations(GDMSstations):
    config = seisblue.utils.Config()
    hyp_file = os.path.join(config.geom, GDMSstations)
    geom = {}
    f = open(hyp_file, 'r')
    for line in f:
        line.strip().strip(',')
        sta = line.strip().split(',')[1][0:4]
        lon = float(line.strip().split(',')[3])
        lat = float(line.strip().split(',')[2])
        elev = float(line.strip().split(',')[4])
        net = line.strip().split(',')[0]

        lat = inventory.util.Latitude(lat)
        lon = inventory.util.Longitude(lon)

        location = {'latitude': lat,
                    'longitude': lon,
                    'elevation': elev}

        geom[sta] = {'network': net,
                     'location': location}
    print(f'read {len(geom)} stations from {GDMSstations}')
    return geom


def write_hyp_station(geom, save_file):
    """
    Write STATION0.HYP file from geometry.

    :param dict geom: Geometry dict.
    :param str save_file: Name of .HYP file.
    """
    config = seisblue.utils.Config()
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

    output = os.path.join(config.geom, save_file)
    with open(output, 'w') as f:
        f.writelines(hyp)


def read_kml_placemark(kml):
    """
    Returns geometry from Google Earth KML file.

    :param str kml: KML file name without directory.
    :rtype: dict
    :return: Geometry dict.
    """
    config = seisblue.utils.Config()
    kml_file = os.path.join(config.geom, kml)

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
        geom[sta] = {'location': location}

    print(f'read {len(geom)} stations from {kml}')
    return geom


def associate_to_txt(txt_path, assciate_list):
    txt_file = open(txt_path, 'w')
    for assoc in assciate_list:
        for event in assoc:
            txt_file.write(
                f'{obspy.UTCDateTime(event.ot)} {event.longitude:7.3f}{event.latitude:7.3f} {event.depth:7.1f} {event.nsta}\n')


def read_header(header):
    header_info = {}
    header_info['year'] = int(header[1:5])
    header_info['month'] = int(header[5:7])
    header_info['day'] = int(header[7:9])
    header_info['hour'] = int(header[9:11])
    header_info['minute'] = int(header[11:13])
    header_info['second'] = float(header[13:19])
    header_info['lat'] = float(header[19:21])
    header_info['lat_minute'] = float(header[21:26])
    header_info['lon'] = int(header[26:29])
    header_info['lon_minute'] = float(header[29:34])
    header_info['depth'] = float(header[34:40]) * 1000
    header_info['magnitude'] = float(header[40:44])
    header_info['nsta'] = header[44:46].replace(" ", "")
    header_info['Pfilename'] = header[46:58].replace(" ", "")
    header_info['newNoPick'] = header[60:63].replace(" ", "")

    return header_info


def read_lines(lines):
    trace = []
    f = open(lines, 'r')
    event_for_each_station = pd.read_fwf(f, header=None, skiprows=1,
                                         colspecs=[(1, 5), (5, 11),
                                                   (12, 15), (16, 19), (19, 20),
                                                   (21, 23), (23, 29), (29, 34),
                                                   (35, 39), (39, 45),
                                                   (45, 50), (51, 55),
                                                   (55, 60), (61, 65), (66, 70),
                                                   (71, 75),
                                                   (76, 77), (78, 83)])
    event_for_each_station.columns = ['station_name', 'epicentral_distance',
                                      'azimuth', 'take_off_angle', 'polarity',
                                      'minute', 'p_arrival_time', 'p_residual',
                                      'p_weight', 's_arrival_time',
                                      's_residual', 's_weight',
                                      'log_A_of_S13', 'log_A_of_A900A',
                                      'ML_of_S13', 'ML_of_A900A', 'intensity',
                                      'PGA']
    for i in range(len(event_for_each_station)):
        trace.append(event_for_each_station.iloc[i].to_dict())

    return trace


def read_afile(afile_path):
    count = 0
    f = open(afile_path, 'r')
    print(afile_path)
    header = f.readline()
    header_info = read_header(header)
    if int(header_info['nsta'])<3:
        return [],0
    trace_info = read_lines(afile_path)
    ev = obspy.core.event.Event()
    ev.event_descriptions.append(obspy.core.event.EventDescription())
    event_time = obspy.UTCDateTime(header_info['year'], header_info['month'],
                                   header_info['day'], header_info['hour'],
                                   header_info['minute'], header_info['second'])

    ev.origins.append(obspy.core.event.Origin(
        time=event_time,
        latitude=header_info['lat'] + header_info['lat_minute'] / 60,
        longitude=header_info['lon'] + header_info['lon_minute'] / 60,
        depth=header_info['depth']))
    for trace in trace_info:
        if len(trace['station_name']) > 4:
            trace['station_name'] = change_station_code(trace['station_name'])
        _waveform_id_1 = obspy.core.event.WaveformStreamID(
            station_code=trace['station_name'],
            channel_code='',
            network_code=''
        )
        for phase in ['P', 'S']:
            try:
                if trace[f'{phase.lower()}_arrival_time'] and  trace[f'{phase.lower()}_residual'] ==0 :
                    continue
                pick_time = obspy.UTCDateTime(header_info['year'],
                                              header_info['month'],
                                              header_info['day'],
                                              header_info['hour'],
                                              int(trace['minute'])) + trace[
                                f'{phase.lower()}_arrival_time']
                if float(trace[f'{phase.lower()}_arrival_time']) != 0:
                    time_errors = obspy.core.event.base.QuantityError()
                    time_errors.confidence_level = trace[
                        f'{phase.lower()}_weight']
                    ev.picks.append(
                        obspy.core.event.Pick(
                            waveform_id=_waveform_id_1,
                            phase_hint=phase,
                            time=pick_time,
                            time_errors=time_errors

                        )
                    )

                    count += 1
            except TypeError:
                print(afile_path,
                      '--------------',
                      'afile_path')
    ev.magnitudes = header_info['magnitude']
    return ev, count


def change_station_code(station):
    if station[0:3] == 'TAP':
        station = 'A' + station[3:6]
    if station[0:3] == 'TCU':
        station = 'B' + station[3:6]
    if station[0:3] == 'CHY':
        station = 'C' + station[3:6]
    if station[0:3] == 'KAU':
        station = 'D' + station[3:6]
    if station[0:3] == 'ILA':
        station = 'E' + station[3:6]
    if station[0:3] == 'HWA':
        station = 'G' + station[3:6]
    if station[0:3] == 'TTN':
        station = 'H' + station[3:6]

    return station


def read_afile_directory(path_list):
    event_list = []
    trace_count = 0
    abs_path = seisblue.utils.get_dir_list(path_list)
    for path in abs_path:
        try:
            event, c = read_afile(path)
            if event:
                event_list.append(event)
                trace_count += c
        except ValueError:
            print('error')
            continue
    print('total_pick = ', trace_count)
    return event_list


def read_CVA_waveform(CVA_list, save_dir):
    progbar = Progbar(len(CVA_list))
    for q, list in enumerate(CVA_list):
        f = open(list, 'r')
        lines = f.readlines()
        count = 0
        t = []
        e = []
        n = []
        z = []
        filename = str(list[-24:-4]).replace('/', '')
        if os.path.exists(f'{save_dir}{filename}.mseed'):
            progbar.add(1)
            continue
        try:
            for line in lines:
                if count == 0:
                    station_code = line.strip()[14:20]
                    if len(station_code) > 4:
                        station_code = change_station_code(station_code)
                        network = 'TSMIP'
                    else:
                        network = 'CWBSN'
                elif count == 2:
                    start_time = obspy.UTCDateTime(line.strip()[12:-1])
                elif count == 3:
                    duration = float(line.strip()[20:28])
                elif count == 4:
                    samplerate = int(line.strip()[17:20])
                elif count == 1:
                    pass
                elif count == 5:
                    pass
                elif count == 6:
                    pass
                elif count == 7:
                    pass
                elif count == 8:
                    pass
                elif count == 9:
                    pass
                elif count == 10:
                    pass
                else:
                    try:
                        tt = float(line[0:10])
                    except ValueError:
                        tt = 0
                    try:
                        te = float(line[10:20])
                    except ValueError:
                        te = 0
                    try:
                        tn = float(line[20:30])
                    except ValueError:
                        tn = 0
                    try:
                        tz = float(line[30:40])
                    except ValueError:
                        tz = 0
                    t.append(tt)
                    e.append(te)
                    n.append(tn)
                    z.append(tz)
                count = count + 1

            traceE = obspy.core.trace.Trace(np.array(e))
            traceN = obspy.core.trace.Trace(np.array(n))
            traceZ = obspy.core.trace.Trace(np.array(z))

            for i, trace in enumerate([traceE, traceN, traceZ]):
                try:
                    trace.stats.network = network
                    trace.stats.station = station_code
                    trace.stats.starttime = start_time
                    trace.stats.sampling_rate = samplerate
                    trace.stats.npts = len(t)
                    trace.stats.delta = t[1] - t[0]
                except IndexError:
                    print(list)
                if i == 0:
                    trace.stats.channel = 'EHE'
                if i == 1:
                    trace.stats.channel = 'EHN'
                if i == 2:
                    trace.stats.channel = 'EHZ'

            st = obspy.core.stream.Stream([traceE, traceN, traceZ])
            progbar.add(1)

            st.write(f'{save_dir}{filename}.mseed',
                     format='MSEED')
            f.close()
        except ValueError:
            print(list)


def associate_to_sfile(associate, database, out_dir):
    picks = seisblue.sql.pick_assoc_id(database=database,
                                       assoc_id=associate.id)
    try:
        seisblue.io.output_sfile(associate.origin_time,
                                 associate.latitude,
                                 associate.longitude,
                                 associate.depth,
                                 picks,
                                 out_dir=out_dir)
    except:
        print()


def output_sfile(ot, latitude, longitude, depth, picks, out_dir,
                 channel_type='EH',
                 filename=None):
    ev = Event()
    ev.event_descriptions.append(EventDescription())
    ev.origins.append(
        Origin(time=UTCDateTime(ot),
               latitude=latitude,
               longitude=longitude,
               depth=depth)
    )

    station = []
    ARC = []
    duration_time = 30
    for pick in picks:
        channel = [f'{channel_type}E', f'{channel_type}N',
                   f'{channel_type}Z']
        network = ''
        location = ''
        if pick.sta not in station:
            station.append(pick.sta)
            for chan in channel:
                ARC.append(
                    f'ARC {pick.sta:<5} {chan:<3} {network:<2} {location:<2} {ot.year:<4} '
                    f'{ot.month:0>2}{ot.day:0>2} {ot.hour:0>2}'
                    f'{ot.minute:0>2} {ot.second:0>2} {duration_time}')

        _waveform_id_1 = WaveformStreamID(
            station_code=pick.sta,
            channel_code=f'{channel_type}Z',
            network_code=pick.net

        )
        # obspy issue #2848 pick.second = 0. bug
        time = UTCDateTime(pick.time)
        if time.second == 0 and time.microsecond == 0:
            time = time + 0.01

        ev.picks.append(
            Pick(waveform_id=_waveform_id_1,
                 phase_hint=pick.phase,
                 time=time,
                 evaluation_mode="automatic"
                 )
        )
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not ARC == []:
            _write_nordic(ev, filename=filename, wavefiles=ARC,
                          outdir=out_dir)


def convert_lat_lon(lat, lon):
    NS = 1
    if lat[-1] == 'S':
        NS = -1

    EW = 1
    if lon[-1] == 'W':
        EW = -1
    if '.' in lat[0:4]:
        lat_degree = int(lat[0:1])
        lat_minute = float(lat[1:-1]) / 60
    else:
        lat_degree = int(lat[0:2])
        lat_minute = float(lat[2:-1]) / 60
    if '.' not in lat:  # high accuracy lat-lon
        lat_minute /= 1000
    lat = (lat_degree + lat_minute) * NS
    lat = inventory.util.Latitude(lat)

    lon_degree = int(lon[0:3])
    lon_minute = float(lon[3:-1]) / 60
    if '.' not in lon:  # high accuracy lat-lon
        lon_minute /= 1000
    lon = (lon_degree + lon_minute) * EW
    lon = inventory.util.Longitude(lon)
    return lat, lon,


def read_hypout(hypo_result):
    picks = []
    azm = []
    origin = obspy.core.event.origin.Origin()
    origin['quality'] = obspy.core.event.origin.OriginQuality()
    origin['origin_uncertainty'] = obspy.core.event.origin.OriginUncertainty()
    f = io.BytesIO(hypo_result.stdout)
    f = f.readlines()

    skip = False
    for i, l in enumerate(f):
        if l.decode('ascii')[2:6] == 'date':
            skip = i
    if not skip:
        return origin, picks

    f = io.BytesIO(hypo_result.stdout)
    #  date hrmn   sec      lat      long depth   no m    rms  damp erln erlt erdp
    # 21 420 2358 51.29 2351.54N 121 32.6E   3.9    6 3   0.19 0.000  6.6 13.2574.0

    hyp_event = pd.read_fwf(f, skiprows=skip, nrows=1,
                            colspecs=[(0, 6), (7, 9), (9, 11), (11, 17),
                                      (17, 26), (26, 36), (36, 42),
                                      (42, 47), (47, 49),
                                      (49, 56), (56, 62),
                                      (62, 67), (67, 72), (72, 77)])
    f = io.BytesIO(hypo_result.stdout)
    # stn   dist   azm  ain w phas    calcphs hrmn tsec  t-obs  t-cal    res   wt di
    # SF64     9 218.3 58.6 0 P    A  PN3     2358 54.0   2.75   2.47   0.28 1.00 11
    hyp_pick = pd.read_fwf(f, skiprows=skip + 2, skipfooter=4,
                           colspecs=[(0, 6), (6, 11),
                                     (11, 17), (17, 22), (22, 24),
                                     (25, 29), (30, 31), (33, 40),
                                     (41, 45), (46, 50),
                                     (50, 57), (57, 64),
                                     (64, 71), (71, 76), (76, 79)])
    for index, row in hyp_event.iterrows():
        lat, lon = convert_lat_lon(row['lat'], row['long'])
        origin.depth = row['depth'] * 1000
        origin.depth_errors.uncertainty = row['erdp']
        origin.latitude = lat
        origin.latitude_errors.uncertainty = row['erlt']
        origin.longitude = lon
        origin.longitude_errors.uncertainty = row['erln']
        origin.quality.used_station_count = row['no']
        origin.quality.azimuthal_gap = 0
        origin.time_errors = row['rms']
        if float(row['sec']) == 60:
            row['mn'] = int(str(row['mn']).strip()) + 1
            row['sec'] = 0
        if float(row['mn']) == 60:
            row['hr'] = int(str(row['hr']).strip()) + 1
            row['mn'] = 0

        origin.time = obspy.UTCDateTime(int('20' + row['date'][0:2].strip()),
                                        int(row['date'][2:4].strip()),
                                        int(row['date'][4:6].strip()),
                                        int(str(row['hr']).strip()),
                                        int(str(row['mn']).strip()),
                                        float(row['sec']))
    for index, row in hyp_pick.iterrows():
        try:
            pick = Pick(waveform_id=WaveformStreamID(station_code=row['stn']),
                        phase_hint=row['phas'],
                        time=origin.time + float(row['t-obs']),
                        evaluation_mode="automatic",
                        time_errors=row['res'])

            arrival = obspy.core.event.origin.Arrival(pick_id=pick.resource_id,
                                                      phase=row['phas'],
                                                      time_residual=row['res'])
        except Exception as err:

            continue
        picks.append(pick)
        origin.arrivals.append(arrival)
        azm.append(float(row['azm']))
    azm.sort(reverse=True)
    try:
        for i in range(len(azm)):
            if i == 0:
                gap = int(abs(azm[i] - (azm[i - 1] + 360)))
            else:
                gap = int(abs(azm[i] - azm[i - 1]))
            if gap > origin.quality.azimuthal_gap:
                origin.quality.azimuthal_gap = gap
    except TypeError as err:
        print(err)
        print(azm[i], '   ', azm[i - 1])
    return origin, picks


if __name__ == "__main__":
    pass
