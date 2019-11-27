import fnmatch
import os
import pickle
import shutil
import yaml
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy
from obspy import read, read_events
from obspy.clients.filesystem.sds import Client
from obspy.core.event.catalog import Catalog
from obspy.core.inventory import Channel, Inventory, Network, Station
from obspy.core.inventory.util import Distance, Latitude, Longitude
from tqdm import tqdm

from seisnn.pick import get_exist_picks, get_pdf, get_pick_list
from seisnn.signal import signal_preprocessing, trim_trace


def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]

def read_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.full_load(file)
    return config

def read_pkl(pkl):
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
        return obj


def write_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_tfrecord():
    pass


def write_tfrecord(output_dir, file_name):
    import tensorflow as tf
    from seisnn.example_proto import trace_to_example

    save_file = os.path.join(output_dir, '{}.tfrecord'.format(file_name))
    with tf.io.TFRecordWriter(save_file) as writer:
        tf_example = trace_to_example(save_file)
        writer.write(tf_example)


def parallel(par, file_list, batch_size=100):
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in tqdm(pool.imap_unordered(par, batch(file_list, batch_size)),
                  total=int(np.ceil(len(file_list) / batch_size))):
        pass

    pool.close()
    pool.join()


def get_dir_list(file_dir, suffix=""):
    file_list = []
    for file_name in os.listdir(file_dir):
        f = os.path.join(file_dir, file_name)
        if file_name.endswith(suffix):
            file_list.append(f)

    return file_list


def read_event_list(file_list):
    catalog = Catalog()
    try:
        for file in file_list:
            catalog += read_events(file)

    except Exception as err:
        print(err)

    catalog.events.sort(key=lambda event: event.origins[0].time)
    return catalog


def read_sds(pick, sds_root, phase="P", component="Z", trace_length=30, sample_rate=100):
    client = Client(sds_root=sds_root)
    if not pick.phase_hint == phase:
        return

    if not pick.waveform_id.channel_code[-1] == component:
        return

    t = pick.time
    t = t - trace_length + np.random.random_sample() * trace_length

    net = "*"
    sta = pick.waveform_id.station_code
    loc = "*"
    chan = "??" + component

    st = client.get_waveforms(net, sta, loc, chan, t, t + trace_length + 1)

    if st.traces:
        trace = st.traces[0]
        trace = signal_preprocessing(trace)
        points = trace_length * sample_rate + 1
        trim_trace(trace, points)
        return trace


def write_training_dataset(catalog, sds_root, output_dir, batch_size=100, remove_dir=False):
    if remove_dir:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    pick_list = get_pick_list(catalog)

    par = partial(_write_picked_trace, pick_list=pick_list, sds_root=sds_root, dataset_dir=output_dir)
    parallel(par, pick_list, batch_size)


def _write_picked_trace(batch_picks, pick_list, sds_root, dataset_dir):
    for pick in batch_picks:
        scipy.random.seed()
        trace = read_sds(pick, sds_root)
        if not trace:
            continue
        exist_picks = get_exist_picks(trace, pick_list)
        trace.picks = exist_picks
        trace.pdf = get_pdf(trace)
        time_stamp = trace.stats.starttime.isoformat()
        trace.write(dataset_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")


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


def read_hyp_inventory(hyp, network, kml_output_dir=None):
    inventory = Inventory(networks=[], source="")
    net = Network(code=network, stations=[], description="")

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
                station = line[1:6]

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

                sta = Station(code=station, latitude=lat, longitude=lon, elevation=elev)

                net.stations.append(sta)

    inventory.networks.append(net)

    if kml_output_dir:
        os.makedirs(kml_output_dir, exist_ok=True)
        inventory.write(kml_output_dir + "/" + network + ".kml", format="KML")

    return inventory


def write_channel_coordinates(dataset_list, dataset_output_dir, inventory, kml_output_dir=None,
                              remove_dataset_dir=False):
    if remove_dataset_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    for i, file in enumerate(dataset_list):
        trace = read(file).traces[0]
        network = trace.stats.network
        station = trace.stats.station
        channel = trace.stats.channel
        location = trace.stats.location

        for net in inventory:
            if not fnmatch.fnmatch(net.code, network):
                continue

            for sta in net:
                if not fnmatch.fnmatch(sta.code, station):
                    continue

                lat = sta.latitude
                lon = sta.longitude
                elev = sta.elevation
                depth = Distance(0)

                trace.stats.coordinates = ({'latitude': lat, 'longitude': lon})
                trace.stats.elevation = elev
                trace.stats.depth = depth

                time_stamp = trace.stats.starttime.isoformat()
                trace.write(dataset_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")
                print(trace.get_id() + " latitude: " + str(lat)[0:5] + " longitude: " + str(lon)[0:6])

                ch_name = []
                for ch in sta.channels:
                    ch_name.append(ch.code)

                if channel not in ch_name:
                    chan = Channel(code=channel, location_code=location, latitude=lat, longitude=lon,
                                   elevation=elev, depth=depth)
                    sta.channels.append(chan)

    if kml_output_dir:
        os.makedirs(kml_output_dir, exist_ok=True)
        inventory.write(kml_output_dir + "/" + inventory.networks[0].code + ".kml", format="KML")
