import obspy.io.nordic.core as nordic
from obspy.core import Stream, read
from obspy.core.event.catalog import Catalog
from obspy.signal.trigger import trigger_onset
import matplotlib.pyplot as plt


def read_list(sfilelist):
    with open(sfilelist, "r") as file:
        data = []
        while True:
            row = file.readline().strip("\n")
            if len(row) == 0:
                break
            data.append(row)
    return data


def load_sfile(sfile_list):
    from obspy.io.nordic.core import NordicParsingError
    catalog = Catalog()
    for sfile in sfile_list:
        try:
            sfile_event, wavename = nordic.read_nordic(sfile, return_wavnames=True)
            sfile_event[0].wavename = wavename
            catalog += sfile_event
        except NordicParsingError:
            pass
    return catalog


def load_seisan(event):
    stream = Stream()
    wavedir = event.wavedir
    for wave in event.wavename:
        stream += read(wavedir + "/" + str(wave))
    stream.sort(keys=['network', 'station', 'channel'])
    stream.normalize()
    stream.detrend()
    return stream


def load_sds(event, sds_root):
    from obspy.clients.filesystem.sds import Client
    t = event.origins[0].time
    stream = Stream()
    client = Client(sds_root=sds_root)
    for pick in event.picks:
        if pick.time > t + 30:
            continue
        if not pick.phase_hint == "P":
            continue
        if not pick.waveform_id.channel_code[-1] == "Z":
            continue
        net = "*"
        sta = pick.waveform_id.station_code
        loc = "*"
        chan = "??" + pick.waveform_id.channel_code[-1]
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


def get_probability(stream):
    import scipy.stats as ss
    for trace in stream:
        start_time = trace.meta.starttime
        x_time = trace.times(reftime=start_time)
        pick_time = trace.pick.time - start_time
        sigma = 0.1
        pdf = ss.norm.pdf(x_time, pick_time, sigma)
        if pdf.max():
            trace.pick.pdf = pdf / pdf.max()
        else:
            stream.remove(trace)
    return stream


def plot_trace(trace):
    start_time = trace.stats.starttime
    pick_time = trace.pick.time - start_time
    pick_phase = trace.pick.phase_hint
    subplot = 2
    fig = plt.figure(figsize=(8, subplot * 2))

    ax = fig.add_subplot(subplot, 1, 1)
    ax.plot(trace.times(reftime=start_time), trace.data, "k-", label=trace.id)
    y_min, y_max = ax.get_ylim()
    ax.vlines(pick_time, y_min, y_max, color='r', lw=2, label=pick_phase)
    ax.legend()

    ax = fig.add_subplot(subplot, 1, subplot)
    ax.plot(trace.times(reftime=start_time), trace.pick.pdf, "b-", label=pick_phase + " pdf")
    ax.legend()
    plt.show()


def plot_stream(stream):
    for trace in stream:
        plot_trace(trace)


def load_dataset(sfile_list, sds_root=None, plot=False):
    catalog = load_sfile(sfile_list)
    dataset = Stream()
    for event in catalog:
        stream = load_sds(event, sds_root)
        stream = get_probability(stream)
        dataset += stream
        if plot:
            plot_stream(stream)
    return dataset


def load_training_set(dataset):
    import numpy as np
    wavefile = []
    probability = []
    for trace in dataset:
        wavefile.append(trace.data)
        probability.append(trace.pick.pdf)

    wavefile = np.asarray(wavefile).reshape((dataset.count(), 1, 3001, 1))
    probability = np.asarray(probability).reshape((dataset.count(), 1, 3001, 1))

    return wavefile, probability


def add_predict(stream, predict):
    i = 0
    for trace in stream:
        trace.pick.pdf = predict[i, :]
        i += 1
    return stream
