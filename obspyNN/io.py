import os
import shutil
from multiprocessing import Pool
from functools import partial

from obspy import read_events
from obspy.core import Stream
from obspy.core.event.catalog import Catalog
from obspy.core.inventory import Inventory, Network, Station, Channel
from obspy.core.inventory.util import Latitude, Longitude
from obspy.clients.filesystem.sds import Client

from obspyNN.pick import get_probability, search_exist_picks, get_pick_list
from obspyNN.signal import signal_preprocessing, trim_trace


def get_dir_list(file_dir, limit=None):
    file_list = []
    for file in list_generator(file_dir):
        file_list.append(os.path.join(file_dir, file))
        if limit and len(file_list) >= limit:
            break

    return file_list


def list_generator(path):
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


def read_sds(event, sds_root, phase="P", component="Z", trace_length=30, sample_rate=100):
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


def write_training_pkl(event_list, sds_root, pkl_dir, remove_dir=False):
    if remove_dir:
        shutil.rmtree(pkl_dir, ignore_errors=True)
    os.makedirs(pkl_dir, exist_ok=True)

    catalog = read_event_list(event_list)
    pick_list = get_pick_list(catalog)

    with Pool() as pool:
        par = partial(_write_picked_trace, pick_list=pick_list, sds_root=sds_root, pkl_dir=pkl_dir)
        pool.map_async(par, catalog.events)
        pool.close()
        pool.join()


def _write_picked_trace(event, pick_list, sds_root, pkl_dir):
    t = event.origins[0].time
    stream = read_sds(event, sds_root)
    for trace in stream:
        picks = search_exist_picks(trace, pick_list)
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
                time_stamp = trace.stats.starttime.isoformat()
                trace.write(pkl_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")

        t += trace_length


def read_hyp_inventory(hyp, network):
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
                chan = Channel(code="??Z", location_code="", latitude=lat, longitude=lon, elevation=elev, depth=0)

                sta.channels.append(chan)
                net.stations.append(sta)

    inventory.networks.append(net)

    return inventory


