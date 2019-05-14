import fnmatch
import os
import pickle
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from obspy import read, read_events
from obspy.clients.filesystem.sds import Client
from obspy.core import Stream
from obspy.core.event.catalog import Catalog
from obspy.core.inventory import Channel, Inventory, Network, Station
from obspy.core.inventory.util import Distance, Latitude, Longitude

from seisnn.pick import get_exist_picks, get_pick_list, get_probability
from seisnn.signal import signal_preprocessing, trim_trace


def read_pickle(pkl):
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
        return obj


def write_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_dir_list(file_dir, limit=None):
    file_list = []
    for file in _list_generator(file_dir):
        file_list.append(os.path.join(file_dir, file))

        if limit and len(file_list) >= limit:
            break

    return file_list


def _list_generator(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def read_event_list(file_list):
    catalog = Catalog()
    for file in file_list:
        catalog += _read_event(file)

    catalog.events.sort(key=lambda event: event.origins[0].time)

    return catalog


def _read_event(event_file):
    catalog = Catalog()
    try:
        cat = read_events(event_file)
        for event in cat.events:
            event.file_name = event_file

        catalog += cat

    except Exception as err:
        print(err)

    return catalog


def read_sds(event, sds_root, phase="P", component="Z", trace_length=30, sample_rate=100, random_time=0):
    stream = Stream()
    client = Client(sds_root=sds_root)
    for pick in event.picks:
        if not pick.phase_hint == phase:
            print("Skip " + pick.phase_hint + " phase pick")
            continue

        if not pick.waveform_id.channel_code[-1] == component:
            print(pick.waveform_id.channel_code)
            continue

        t = event.origins[0].time

        if pick.time > t + trace_length:
            t = pick.time - trace_length + 5
            print("origin: " + t.isoformat() + " pick: " + pick.time.isoformat())

        if random_time:
            t = t - random_time + np.random.random_sample() * random_time * 2

        net = "*"
        sta = pick.waveform_id.station_code
        loc = "*"
        chan = "??" + component

        st = client.get_waveforms(net, sta, loc, chan, t, t + trace_length + 1)

        if st.traces:
            trace = st.traces[0]
            trace = signal_preprocessing(trace)

            points = trace_length * sample_rate + 1
            try:
                trim_trace(trace, points)

            except Exception as err:
                print(err)
                continue

            stream += st.traces[0]

        else:
            print("No trace in ", t.isoformat(), net, sta, loc, chan)

    return stream


def write_training_pkl(catalog, sds_root, pkl_dir, random_time=0, remove_dir=False):
    if remove_dir:
        shutil.rmtree(pkl_dir, ignore_errors=True)
    os.makedirs(pkl_dir, exist_ok=True)

    pick_list = get_pick_list(catalog)
    pool_size = cpu_count()

    with Pool(processes=pool_size, maxtasksperchild=1) as pool:
        par = partial(_write_picked_trace, pick_list=pick_list, sds_root=sds_root, pkl_dir=pkl_dir,
                      random_time=random_time)
        pool.map_async(par, catalog.events)
        pool.close()
        pool.join()


def _write_picked_trace(event, pick_list, sds_root, pkl_dir, random_time):
    t = event.origins[0].time
    stream = read_sds(event, sds_root, random_time)
    for trace in stream:
        picks = get_exist_picks(trace, pick_list)
        trace.picks = picks
        trace.pdf = get_probability(trace)
        time_stamp = trace.stats.starttime.isoformat()
        trace.write(pkl_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")

    print(event.file_name + " " + t.isoformat() + " load " + str(len(stream)) + " traces, total "
          + str(len(pick_list)) + " picks")


def write_station_pkl(pkl_output_dir, sds_root, nslc, start_time, end_time,
                      trace_length=30, sample_rate=100, remove_dir=False):
    if remove_dir:
        shutil.rmtree(pkl_output_dir, ignore_errors=True)
    os.makedirs(pkl_output_dir, exist_ok=True)

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
                trace.write(pkl_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")
                counter += 1

                if counter % 100 == 0:
                    print("Output file... %d" % counter)

        t += trace_length
    print("Output file... Total %d" % counter)


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


def write_channel_coordinates(pkl_list, pkl_output_dir, inventory, kml_output_dir=None, remove_pkl_dir=False):
    if remove_pkl_dir:
        shutil.rmtree(pkl_output_dir, ignore_errors=True)
    os.makedirs(pkl_output_dir, exist_ok=True)

    for i, file in enumerate(pkl_list):
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
                trace.write(pkl_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")
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
