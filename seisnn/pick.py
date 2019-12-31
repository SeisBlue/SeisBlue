from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks

from obspy import read

import tensorflow as tf
import tensorflow_probability as tfp

from seisnn.utils import binary_search


def get_pick_dict(events):
    pick_dict = {}
    pick_count = {}
    for event in events:
        for p in event.picks:
            station = p.waveform_id.station_code
            if not pick_dict.get(station):
                pick_dict[station] = []
            phase = p.phase_hint
            if not pick_count.get(phase):
                pick_count[phase] = 0
            pick_dict[station].append(p)
            pick_count[phase] += 1

    pick_dict = OrderedDict(sorted(pick_dict.items()))
    for k, v in pick_dict.items():
        v.sort(key=lambda pick: pick.time)
        print(f'station {k} {len(v)} picks')

    for k, v in pick_count.items():
        print(f'total {v} {k} phase picks')
    return pick_dict


def get_window(pick, trace_length=30):
    scipy.random.seed()
    pick_time = pick.time

    starttime = pick_time - trace_length + np.random.random_sample() * trace_length
    endtime = starttime + trace_length

    window = {
        'starttime': starttime,
        'endtime': endtime,
        'station': pick.waveform_id.station_code,
        'wavename': pick.waveform_id.wavename
    }
    return window


def get_pdf(stream, sigma=0.1):
    trace = stream[0]
    tfd = tfp.distributions
    start_time = trace.stats.starttime
    x_time = trace.times(reftime=start_time)
    stream.pdf = {}
    for phase, picks in stream.picks.items():
        phase_pdf = np.zeros((len(x_time),))

        for pick in picks:
            pick_time = pick.time - start_time
            dist = tfd.Normal(loc=pick_time, scale=sigma)
            pick_pdf = dist.prob(x_time)

            if tf.math.reduce_max(pick_pdf):
                phase_pdf += pick_pdf / tf.math.reduce_max(pick_pdf)

        if tf.math.reduce_max(phase_pdf):
            phase_pdf = phase_pdf / tf.math.reduce_max(phase_pdf)

        stream.pdf[phase] = phase_pdf.numpy()
    return stream

def get_picks_from_pdf(feature, phase_type, height=0.5, distance=100):
    start_time = feature.starttime
    peaks, properties = find_peaks(feature.phase[phase_type], height=height, distance=distance)

    pick_time = []
    pick_phase = []
    pick_set = []
    for p in peaks:
        if p:
            pick_time.append(start_time + p * feature.delta)
            pick_phase.append('P')
            pick_set.append(phase_type)

    picks = {'pick_time': pick_time,
             'pick_phase': pick_phase,
             'pick_set': pick_set}

    picks = pd.DataFrame.from_dict(picks)

    return picks


def get_picks_from_dataset(dataset):
    pick_list = []
    trace = read(dataset, headonly=True).traces[0]
    picks = trace.picks
    pick_list.extend(picks)
    return pick_list


def search_pick(pick_list, pick_time_key, stream):
    tmp_pick = {}
    starttime = stream.traces[0].stats.starttime
    endtime = stream.traces[0].stats.endtime
    station = stream.traces[0].stats.station

    left, right = binary_search(pick_time_key, starttime, endtime)

    for pick in pick_list[left:right]:
        if not pick.waveform_id.station_code == station:
            continue
        phase = pick.phase_hint
        if not tmp_pick.get(phase):
            tmp_pick[phase] = [pick]
        else:
            tmp_pick[phase].append(pick)

    return tmp_pick


def validate_picks_nearby(validate_pick, predict_pick, delta=0.1):
    pick_upper_bound = predict_pick.time + delta
    pick_lower_bound = predict_pick.time - delta
    if pick_lower_bound < validate_pick.time < pick_upper_bound:
        return True
    else:
        return False


def get_time_residual(pre_pick, val_pick):
    if validate_picks_nearby(pre_pick, val_pick, delta=0.5):
        residual = val_pick.time - pre_pick.time
        return residual
