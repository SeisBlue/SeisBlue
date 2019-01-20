import fnmatch
import numpy as np
import scipy.stats as ss
from scipy.signal import find_peaks
from bisect import bisect_left, bisect_right

from obspy import read
from obspy.core.event.origin import Pick


def get_pick_list(catalog):
    pick_list = []
    for event in catalog:
        for pick in event.picks:
            pick_list.append(pick)

    pick_list.sort(key=lambda pick: pick.time)
    print("total " + str(len(pick_list)) + " picks")

    return pick_list


def get_probability(trace, sigma=0.1):
    start_time = trace.stats.starttime
    x_time = trace.times(reftime=start_time)
    pdf = np.zeros((len(x_time),))

    for pick in trace.picks:
        pick_time = pick.time - start_time
        pick_pdf = ss.norm.pdf(x_time, pick_time, sigma)

        if pick_pdf.max():
            pdf += pick_pdf / pick_pdf.max()

    if pdf.max():
        pdf = pdf / pdf.max()

    return pdf


def set_probability(predict, pkl_list, pkl_output_dir):
    for i, prob in enumerate(predict):
        try:
            trace = read(pkl_list[i]).traces[0]

        except IndexError:
            break

        trace_length = trace.data.size
        pdf = prob.reshape(trace_length, )

        if pdf.max():
            trace.pdf = pdf / pdf.max()
        else:
            trace.pdf = pdf

        trace.picks = get_picks_from_pdf(trace)
        time_stamp = trace.stats.starttime.isoformat()
        trace.write(pkl_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")


def get_picks_from_pdf(trace, height=0.5, width=10):
    start_time = trace.stats.starttime
    peaks, properties = find_peaks(trace.pdf, height=height, width=width)

    picks = []
    for p in peaks:
        if p:
            time = start_time + p / trace.stats.sampling_rate
            phase_hint = "P"
            pick = Pick(time=time, phase_hint=phase_hint)
            picks.append(pick)

    return picks


def filter_pick_time_window(pick_list, start_time, end_time):
    # binary search, pick_list must be sorted by time
    pick_time_key = []
    for pick in pick_list:
        pick_time_key.append(pick.time)

    left = bisect_left(pick_time_key, start_time)
    right = bisect_right(pick_time_key, end_time)
    pick_list = pick_list[left:right]

    return pick_list


def search_exist_picks(trace, pick_list, phase="P"):
    start_time = trace.stats.starttime
    end_time = trace.stats.endtime
    network = trace.stats.network
    station = trace.stats.station
    location = trace.stats.location
    channel = "*" + trace.stats.channel[-1]

    pick_list = filter_pick_time_window(pick_list, start_time, end_time)

    tmp_pick = []
    for pick in pick_list:
        network_code = pick.waveform_id.network_code
        station_code = pick.waveform_id.station_code
        location_code = pick.waveform_id.location_code
        channel_code = pick.waveform_id.channel_code

        if not start_time < pick.time < end_time:
            continue

        if not pick.phase_hint == phase:
            continue

        if network and network_code and not network_code == 'NA':
            if not fnmatch.fnmatch(network_code, network):
                continue

        if station:
            if not fnmatch.fnmatch(station_code, station):
                continue

        if location and location_code:
            if not fnmatch.fnmatch(location_code, location):
                continue

        if channel:
            if not fnmatch.fnmatch(channel_code, channel):
                continue

        tmp_pick.append(pick)

    return tmp_pick
