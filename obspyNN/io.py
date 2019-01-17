import numpy as np

from obspy import read_events
from obspy.core import Stream
from obspy.core.event.catalog import Catalog
from obspy.core.inventory import Inventory, Network, Station, Channel
from obspy.core.inventory.util import Latitude, Longitude
from obspy.clients.filesystem.sds import Client

from obspyNN.plot import plot_trace
from obspyNN.pick import get_probability, search_picks, get_pick_list


class LengthError(BaseException):
    pass


def read_event_list(sfile_list):
    with open(sfile_list, "r") as file:
        catalog = Catalog()
        while True:
            line = file.readline().rstrip()
            if len(line) == 0:
                break
            catalog += _read_event(line)
    catalog.events.sort(key=lambda event: event.time)
    return catalog


def _read_event(event_file):
    catalog = Catalog()
    try:
        sfile_event = read_events(event_file)
        for event in sfile_event.events:
            event.sfile = event_file
        catalog += sfile_event
    except Exception as err:
        print(err)
    return catalog


def data_preprocessing(data):
    data.detrend('demean')
    data.detrend('linear')
    data.normalize()
    data.resample(100)
    return data


def trim_trace_by_points(trace, points=3001):
    start_time = trace.stats.starttime
    dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
    end_time = start_time + dt * (points - 1)
    trace.trim(start_time, end_time, nearest_sample=False, pad=True, fill_value=0)
    if not trace.data.size == points:
        raise LengthError("Trace length is not correct.")


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
        trace = st.traces[0]
        trace = data_preprocessing(trace)

        points = trace_length * sample_rate + 1
        try:
            trim_trace_by_points(trace, points)
        except Exception as err:
            print(err)
            continue

        stream += st.traces[0]

    return stream


def get_picked_stream(sfile_list, sds_root=None, plot=False):
    catalog = read_event_list(sfile_list)
    pick_list = get_pick_list(catalog)
    stream = Stream()
    for event in catalog:
        t = event.origins[0].time
        st = read_sds(event, sds_root)

        for trace in st:
            picks = search_picks(trace, pick_list)
            trace.picks = picks

        st = get_probability(st)
        stream += st
        print(event.sfile + " " + t.isoformat() + " load " + str(len(st)) + " traces, total " + str(
            len(stream)) + " traces")
        if plot:
            plot_trace(st, savedir="original_dataset")
    return stream


def get_training_set(dataset, points=3001):
    wavefile = []
    probability = []
    for trace in dataset:
        wavefile.append(trace.data)
        probability.append(trace.pdf)

    component = 1
    output_shape = (dataset.count(), component, points, 1)

    wavefile = np.asarray(wavefile).reshape(output_shape)
    probability = np.asarray(probability).reshape(output_shape)

    return wavefile, probability


def scan_station(sds_root=None, nslc=None, start_time=None, end_time=None, trace_length=30, sample_rate=100):
    client = Client(sds_root=sds_root)
    stream = Stream()
    net, sta, loc, chan = nslc
    t = start_time
    while t < end_time:
        st = client.get_waveforms(net, sta, loc, chan, t, t + trace_length + 1)
        st = data_preprocessing(st)

        points = trace_length * sample_rate + 1
        for trace in st:
            try:
                trim_trace_by_points(trace, points)
            except IndexError as err:
                print(err)
                st.remove(trace)
                continue

        stream += st
        t += trace_length
    return stream


def read_hyp(hyp, network):
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
