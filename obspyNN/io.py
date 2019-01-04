from obspy.core import Stream
from obspy.core.event.catalog import Catalog
from obspy.core.event.origin import Pick

from obspy.core.inventory import Inventory, Network, Station, Channel
from obspy.core.inventory.util import Latitude, Longitude

import obspy.io.nordic.core as nordic
from obspy.io.nordic.core import NordicParsingError
from obspy.clients.filesystem.sds import Client

import numpy as np

from obspyNN.plot import plot_trace
from obspyNN.probability import get_probability


def read_sfile_list(sfile_list):
    with open(sfile_list, "r") as file:
        data = []
        while True:
            row = file.readline().rstrip()
            if len(row) == 0:
                break
            data.append(row)
    return data


def load_sfile(sfile_list):
    catalog = Catalog()
    for sfile in sfile_list:
        try:
            sfile_event = nordic.read_nordic(sfile)
            sfile_event.events[0].sfile = sfile
            catalog += sfile_event
        except NordicParsingError:
            pass
    return catalog


def load_sds(event, sds_root, phase="P", component="Z"):
    stream = Stream()
    client = Client(sds_root=sds_root)
    for pick in event.picks:
        if not pick.phase_hint == phase:
            continue
        if not pick.waveform_id.channel_code[-1] == component:
            continue

        t = event.origins[0].time
        if pick.time > t + 30:
            t = pick.time - 25
            print("origin: " + t.isoformat() + " pick: " + pick.time.isoformat())

        net = "*"
        sta = pick.waveform_id.station_code
        loc = "*"
        chan = "??" + component

        st = client.get_waveforms(net, sta, loc, chan, t, t + 31)
        st.normalize()
        st.detrend()
        st.resample(100)

        desired_trace_length = 3001
        for trace in st:
            dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
            endtime = t + 3000 * dt
            trace.trim(t, endtime, pad=True, fill_value=0)
            trace.pick = pick

            if not trace.data.size == desired_trace_length:
                print(trace.id, trace.data.size)
                st.remove(trace)

        stream += st
    stream.sort(keys=['starttime', 'network', 'station', 'channel'])
    return stream


def load_stream(sfile_list, sds_root=None, plot=False):
    catalog = load_sfile(sfile_list)
    dataset = Stream()
    for event in catalog:
        stream = load_sds(event, sds_root)
        stream = get_probability(stream)
        dataset += stream
        print(event.sfile + " load " + str(len(stream)) + " traces, total " + str(len(dataset)) + " traces")
        if plot:
            plot_trace(stream, savedir="original_dataset")
    return dataset


def load_training_set(dataset, trace_length=3001):
    wavefile = []
    probability = []
    for trace in dataset:
        wavefile.append(trace.data)
        probability.append(trace.pick.pdf)

    component = 1
    output_shape = (dataset.count(), component, trace_length, 1)

    wavefile = np.asarray(wavefile).reshape(output_shape)
    probability = np.asarray(probability).reshape(output_shape)

    return wavefile, probability


def scan_station(sds_root=None, nslc=None, start_time=None, end_time=None):
    client = Client(sds_root=sds_root)
    stream = Stream()
    net, sta, loc, chan = nslc
    t = start_time
    while t < end_time:
        st = client.get_waveforms(net, sta, loc, chan, t, t + 31)
        st.normalize()
        st.detrend()
        st.resample(100)
        st.trim(t, t + 30, pad=True, fill_value=0)

        for trace in st:
            if trace.data.size == 3001:
                trace.pick = Pick()
                trace.pick.time = t
                trace.pick.phase_hint = "P"
            else:
                st.remove(trace)
        stream += st
        t += 30
    return stream


def read_geom(hyp, network):
    inv = Inventory(networks=[], source="")
    net = Network(code=network, stations=[], description="")
    with open(hyp, 'r') as file:
        stats = 0
        while True:
            line = file.readline().rstrip()
            if not len(line):
                stats += 1
                continue

            if stats > 1:
                break
            elif stats == 1:
                station = line[1:6]
                lat = line[6:14]
                lon = line[14:23]
                elev = float(line[23:])

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
    inv.networks.append(net)
    return inv
