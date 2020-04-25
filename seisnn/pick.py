"""
Pick
=============

.. autosummary::
    :toctree: pick

    get_pdf
    get_picks_from_dataset
    get_picks_from_pdf
    get_time_residual
    get_window
    validate_picks_nearby

"""

import numpy as np
import scipy
import scipy.stats as ss
from scipy.signal import find_peaks

from obspy import read, UTCDateTime


def get_window(pick, trace_length=30):
    scipy.random.seed()
    pick_time = UTCDateTime(pick.time)

    starttime = pick_time - trace_length + np.random.random_sample() * trace_length
    endtime = starttime + trace_length

    window = {
        'starttime': starttime,
        'endtime': endtime,
        'station': pick.station,
    }
    return window


def get_pdf(stream, sigma=0.1):
    trace = stream[0]
    start_time = trace.stats.starttime
    x_time = trace.times(reftime=start_time)

    stream.pdf = np.zeros([3008, 2])
    stream.phase = []

    for i, phase in enumerate(['P', 'S']):
        if stream.picks.get(phase):
            stream.phase.append(phase)
        else:
            stream.phase.append('')
            continue

        phase_pdf = np.zeros((len(x_time),))
        for pick in stream.picks[phase]:
            pick_time = UTCDateTime(pick.time) - start_time
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
