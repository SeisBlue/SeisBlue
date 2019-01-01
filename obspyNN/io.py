from obspy.core import Stream
from obspy.core.event.catalog import Catalog

import obspy.io.nordic.core as nordic
from obspy.io.nordic.core import NordicParsingError
from obspy.clients.filesystem.sds import Client

import numpy as np

from obspyNN.plot import plot_stream
from obspyNN.probability import get_probability


def read_list(sfile_list):
    with open(sfile_list, "r") as file:
        data = []
        while True:
            row = file.readline().strip("\n")
            if len(row) == 0:
                break
            data.append(row)
    return data


def load_sfile(sfile_list):
    catalog = Catalog()
    for sfile in sfile_list:
        try:
            sfile_event = nordic.read_nordic(sfile, return_wavnames=False)
            catalog += sfile_event
        except NordicParsingError:
            pass
    return catalog


def load_sds(event, sds_root, phase="P", component="Z"):
    t = event.origins[0].time
    stream = Stream()
    client = Client(sds_root=sds_root)
    for pick in event.picks:
        if pick.time > t + 30:
            continue
        if not pick.phase_hint == phase:
            continue
        if not pick.waveform_id.channel_code[-1] == component:
            continue

        net = "*"
        sta = pick.waveform_id.station_code
        loc = "*"
        chan = "??" + component

        st = client.get_waveforms(net, sta, loc, chan, t, t + 31)
        st.normalize()
        st.detrend()
        st.resample(100)
        st.trim(t, t + 30, pad=True, fill_value=0)

        for trace in st:
            if trace.data.size == 3001:
                trace.pick = pick
            else:
                st.remove(trace)
        stream += st
    stream.sort(keys=['starttime', 'network', 'station', 'channel'])
    return stream


def get_picked_stream(event, stream):
    picked_stream = Stream()
    for pick in event.picks:
        station = pick.waveform_id.station_code
        station_stream = stream.select(station=station)
        for trace in station_stream:
            trace.pick = pick
        picked_stream += station_stream
    return picked_stream


def load_dataset(sfile_list, sds_root=None, plot=False):
    catalog = load_sfile(sfile_list)
    dataset = Stream()
    for event in catalog:
        stream = load_sds(event, sds_root)
        stream = get_probability(stream)
        dataset += stream
        print("load " + str(len(dataset)) + " traces")
        if plot:
            plot_stream(stream)
    return dataset


def load_training_set(dataset, trace_length=3001):
    wavefile = []
    probability = []
    for trace in dataset:
        wavefile.append(trace.data)
        probability.append(trace.pick.pdf)

    wavefile = np.asarray(wavefile).reshape((dataset.count(), 1, trace_length, 1))
    probability = np.asarray(probability).reshape((dataset.count(), 1, trace_length, 1))

    return wavefile, probability
