import fnmatch
import os
import shutil
from bisect import bisect_left, bisect_right

import numpy as np
import scipy.stats as ss
from obspy import read
from obspy.core.event.base import QuantityError, WaveformStreamID
from obspy.core.event.origin import Pick
from scipy.signal import find_peaks


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


def get_picks_from_pdf(trace, height=0.5, distance=100):
    start_time = trace.stats.starttime
    peaks, properties = find_peaks(trace.pdf, height=height, distance=distance)

    picks = []
    for p in peaks:
        if p:
            time = start_time + p / trace.stats.sampling_rate
            phase_hint = "P"
            pick = Pick(time=time, phase_hint=phase_hint)
            pick.waveform_id = WaveformStreamID(network_code=trace.stats.network, station_code=trace.stats.station,
                                                location_code=trace.stats.channel, channel_code=trace.stats.location)
            picks.append(pick)

    return picks


def get_picks_from_pkl(pkl):
    pick_list = []
    trace = read(pkl, headonly=True).traces[0]
    picks = trace.picks
    pick_list.extend(picks)
    return pick_list


def _filter_pick_time_window(pick_list, start_time, end_time):
    # binary search, pick_list must be sorted by time
    pick_time_key = []
    for pick in pick_list:
        pick_time_key.append(pick.time)

    left = bisect_left(pick_time_key, start_time)
    right = bisect_right(pick_time_key, end_time)
    pick_list = pick_list[left:right]

    return pick_list


def get_exist_picks(trace, pick_list, phase="P"):
    start_time = trace.stats.starttime
    end_time = trace.stats.endtime
    network = trace.stats.network
    station = trace.stats.station
    location = trace.stats.location
    channel = "*" + trace.stats.channel[-1]

    pick_list = _filter_pick_time_window(pick_list, start_time, end_time)

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

        pick.evaluation_mode = "manual"
        tmp_pick.append(pick)

    return tmp_pick


def write_probability_pkl(predict, pkl_list, pkl_output_dir, remove_dir=False):
    if remove_dir:
        shutil.rmtree(pkl_output_dir, ignore_errors=True)
    os.makedirs(pkl_output_dir, exist_ok=True)

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
        pdf_picks = get_picks_from_pdf(trace)

        if trace.picks:
            for val_pick in trace.picks:
                for pre_pick in pdf_picks:
                    pre_pick.evaluation_mode = "automatic"

                    residual = get_time_residual(val_pick, pre_pick)
                    pre_pick.time_errors = QuantityError(residual)

                    if is_close_pick(val_pick, pre_pick, delta=0.1):
                        pre_pick.evaluation_status = "confirmed"
                    elif is_close_pick(val_pick, pre_pick, delta=1):
                        pre_pick.evaluation_status = "rejected"

        else:
            trace.picks = []
            for pre_pick in pdf_picks:
                pre_pick.evaluation_mode = "automatic"

        trace.picks.extend(pdf_picks)
        time_stamp = trace.stats.starttime.isoformat()
        trace.write(pkl_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")

        if i % 1000 == 0:
            print("Output file... %d out of %d" % (i, len(predict)))


def is_close_pick(validate_pick, predict_pick, delta=0.1):
    pick_upper_bound = predict_pick.time + delta
    pick_lower_bound = predict_pick.time - delta
    if pick_lower_bound < validate_pick.time < pick_upper_bound:
        return True
    else:
        return False


def get_time_residual(val_pick, pre_pick):
    if is_close_pick(val_pick, pre_pick, delta=0.5):
        residual = val_pick.time - pre_pick.time
        return residual
    else:
        return None
