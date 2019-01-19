import os
import numpy as np
from tensorflow.python.keras.utils import Sequence
from multiprocessing import Pool
from functools import partial

from obspy import read, read_events
from obspy.core import Stream
from obspy.core.event.catalog import Catalog
from obspy.core.inventory import Inventory, Network, Station, Channel
from obspy.core.inventory.util import Latitude, Longitude
from obspy.clients.filesystem.sds import Client

from obspyNN.pick import get_probability, search_exist_picks, get_pick_list
from obspyNN.signal import signal_preprocessing, trim_trace


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def filecount(dir_name):
    return len([f for f in os.listdir(dir_name) if os.path.isfile(f)])


def read_event_list(sfile_list):
    catalog = Catalog()
    for sfile in sfile_list:
        catalog += _read_event(sfile)
    catalog.events.sort(key=lambda event: event.origins[0].time)
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


def generate_trainning_pkl(sfile_list, sds_root=None, pkl_dir=None):
    catalog = read_event_list(sfile_list)
    pick_list = get_pick_list(catalog)

    with Pool() as pool:
        par = partial(generate_picked_trace, pick_list=pick_list, sds_root=sds_root, pkl_dir=pkl_dir)
        pool.map_async(par, catalog.events)
        pool.close()
        pool.join()


def generate_picked_trace(event, pick_list, sds_root=None, pkl_dir=None):
    t = event.origins[0].time
    stream = read_sds(event, sds_root)
    for trace in stream:
        picks = search_exist_picks(trace, pick_list)
        trace.picks = picks
        trace.pdf = get_probability(trace)
        time_stamp = trace.stats.starttime.isoformat()
        trace.write(pkl_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")

    print(event.sfile + " " + t.isoformat() + " load " + str(len(stream)) + " traces, total "
          + str(len(pick_list)) + " picks")


def get_training_set(stream, shape=(1, 3001, 1)):
    output_shape = shape
    trace = stream.traces[0]
    wavefile = trace.data.reshape(output_shape)
    probability = trace.pdf.reshape(output_shape)
    return wavefile, probability


def generate_station_pkl(pkl_dir, sds_root, nslc, start_time, end_time, trace_length=30, sample_rate=100):
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
                trace.write(pkl_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")

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


class DataGenerator(Sequence):
    def __init__(self, pkl_list, batch_size=2, dim=(1, 3001, 1), shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.pkl_list = pkl_list
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pkl_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_pkl_list = [self.pkl_list[k] for k in indexes]
        wavefile, probability = self.__data_generation(temp_pkl_list)

        return wavefile, probability

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pkl_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_pkl_list):
        wavefile = np.empty((self.batch_size, *self.dim))
        probability = np.empty((self.batch_size, *self.dim))
        for i, ID in enumerate(temp_pkl_list):
            trace = read(ID).traces[0]
            wavefile[i,], probability[i,] = get_training_set(trace, self.dim)

        return wavefile, probability


class PredictGenerator(DataGenerator):
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_pkl_list = [self.pkl_list[k] for k in indexes]
        wavefile = self.__data_generation(temp_pkl_list)

        return wavefile

    def __data_generation(self, temp_pkl_list):
        wavefile = np.empty((self.batch_size, *self.dim))
        for i, ID in enumerate(temp_pkl_list):
            trace = read(ID).traces[0]
            wavefile[i,] = trace.data.reshape(*self.dim)
        return wavefile
