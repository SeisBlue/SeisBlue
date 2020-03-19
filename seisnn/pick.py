"""
Pick
=============

"""

from collections import OrderedDict

import numpy as np
import scipy
import scipy.stats as ss
from scipy.signal import find_peaks

from obspy import read, UTCDateTime
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
    start_time = trace.stats.starttime
    x_time = trace.times(reftime=start_time)

    stream.pdf = np.zeros([3008, 1])
    stream.phase = []

    for i, phase in enumerate(['P']):
        if stream.picks.get(phase):
            stream.phase.append(phase)
        else:
            stream.phase.append('')
            continue

        phase_pdf = np.zeros((len(x_time),))
        for pick in stream.picks[phase]:
            pick_time = pick.time - start_time
            pick_pdf = ss.norm.pdf(x_time, pick_time, sigma)

            if pick_pdf.max():
                phase_pdf += pick_pdf / pick_pdf.max()

        stream.pdf[:, i] = phase_pdf
    return stream


def get_picks_from_pdf(feature, phase, pick_set, height=0.5, distance=100):
    i = feature.phase.index(phase)
    peaks, properties = find_peaks(feature.pdf[-1, :, i], height=height, distance=distance)

    for p in peaks:
        if p:
            pick_time = UTCDateTime(feature.starttime) + p * feature.delta
            feature.pick_time.append(pick_time.isoformat())
            feature.pick_phase.append(feature.phase[i])
            feature.pick_set.append(pick_set)


def get_picks_from_dataset(dataset):
    pick_list = []
    trace = read(dataset, headonly=True).traces[0]
    picks = trace.picks
    pick_list.extend(picks)
    return pick_list


def search_pick(pick_list, pick_time_key, stream):
    tmp_pick = {'P': []}
    starttime = stream.traces[0].stats.starttime
    endtime = stream.traces[0].stats.endtime
    station = stream.traces[0].stats.station

    left, right = binary_search(pick_time_key, starttime, endtime)

    for pick in pick_list[left:right]:
        if not pick.waveform_id.station_code == station:
            continue
        phase = pick.phase_hint
        if phase in tmp_pick:
            tmp_pick[phase].append(pick)

    return tmp_pick


def validate_picks_nearby(val_pick, pred_pick, delta=0.1):
    pick_upper_bound = UTCDateTime(pred_pick['pick_time']) + delta
    pick_lower_bound = UTCDateTime(pred_pick['pick_time']) - delta
    if pick_lower_bound < UTCDateTime(val_pick['pick_time']) < pick_upper_bound:
        return True
    else:
        return False


def get_time_residual(val_pick, pred_pick):
    residual = UTCDateTime(val_pick['pick_time']) - UTCDateTime(pred_pick['pick_time'])
    return residual

if __name__ == "__main__":
    pass
