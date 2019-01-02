from obspy.core import Stream
from obspy.core.event.catalog import Catalog
from obspy.core.event.origin import Pick

import obspy.io.nordic.core as nordic
from obspy.io.nordic.core import NordicParsingError
from obspy.clients.filesystem.sds import Client

import numpy as np

from obspyNN.plot import plot_stream
from obspyNN.probability import get_probability


def read_sfile_list(sfile_list):
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
            sfile_event = nordic.read_nordic(sfile)
            sfile_event.events[0].sfile = sfile
            catalog += sfile_event
        except NordicParsingError:
            pass
    return catalog


def load_sds(event, sds_root, phase="P", component="Z"):
    t = event.origins[0].time
    stream = Stream()
    client = Client(sds_root=sds_root)
    for pick in event.picks:
        if not pick.time < t + 30:
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

        desired_trace_length = 3001
        for trace in st:
            if trace.data.size == desired_trace_length:
                trace.pick = pick
            else:
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
            plot_stream(stream, savedir="original_dataset")
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
