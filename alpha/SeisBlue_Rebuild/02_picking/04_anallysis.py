# -*- coding: utf-8 -*-
import seisblue

import glob
import numpy as np
from datetime import datetime, timedelta
import scipy
import obspy
from itertools import chain


def get_pick_snr(instances):
    p_snr = []
    s_snr = []
    for instance in instances:
        for label in instance.labels:
            try:
                if label.tag == 'manual':
                    label.timewindow = label.timewindow or instance.timewindow
                    get_picks_by_threshold(label)
                    if len(label.picks) > 0:
                        for pick in label.picks:
                            get_snr(pick, instance.features, instance.timewindow)
                            if pick.phase == 'P':
                                p_snr.append(pick.snr)
                            elif pick.phase == 'S':
                                s_snr.append(pick.snr)
            except Exception as e:
                print(e)
    return [p_snr, s_snr]

def get_snr(pick, features, timewindow, second=1):
    try:
        vector = np.linalg.norm(features.data, axis=0)
        point = int((pick.time - timewindow.starttime).total_seconds() * 100)
        if point >= second * 100:
            signal = vector[point: point + second * 100]
            noise = vector[point - len(signal): point]
        else:
            noise = vector[0:point]
            signal = vector[point: point + len(noise)]
        snr = signal_to_noise_ratio(signal=signal, noise=noise)
        pick.snr = np.around(snr, 4)
    except Exception as e:
        print(e)


def signal_to_noise_ratio(signal, noise):
    """
    Calculates power ratio from signal and noise.

    :param numpy.array signal: Signal trace data.
    :param numpy.array noise: Noise trace data.
    :rtype: float
    :return: Signal to noise ratio.
    """
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(noise))
    if noise_power == 0:
        noise_power = np.finfo(float).eps
    power_ratio = signal_power / noise_power
    if power_ratio <= 0:
        power_ratio = np.finfo(float).eps
    snr = np.log10(power_ratio)
    return snr


def get_picks_by_threshold(label, threshold={'P': 0.5, 'S': 0.5}, distance=100,
                           from_ms=0, to_ms=-1):
    """
    Extract pick from label and write into the database.
    :param float height: Height threshold, from 0 to 1, default is 0.5.
    :param int distance: Distance threshold in data point.
    """
    picks = []
    for i, phase in enumerate(label.phase[0:2]):
        peaks, properties = scipy.signal.find_peaks(
            label.data[i, from_ms:to_ms], height=threshold[phase],
            distance=distance
        )

        for j, peak in enumerate(peaks):
            if peak:
                pick_time = (
                        obspy.UTCDateTime(label.timewindow.starttime)
                        + (peak + from_ms) * label.timewindow.delta
                )

                picks.append(
                    seisblue.core.Pick(
                        time=datetime.utcfromtimestamp(pick_time.timestamp),
                        inventory=label.inventory,
                        phase=label.phase[i],
                        tag=label.tag,
                        confidence=round(float(properties["peak_heights"][j]),
                                         2),
                    )
                )
    label.picks = picks


if __name__ == '__main__':
    config = seisblue.utils.get_config("--data_config_filepath")
    c = config['analysis']
    # filepaths_all = list(sorted(glob.glob(c['datasetpath'])))
    # filepaths_par = [filepaths_all[i:i + 100] for i in
    #                 range(0, len(filepaths_all), 100)]
    # pick_snr_p = []
    # pick_snr_s = []
    # for filepaths in filepaths_par:
    #     instances = seisblue.utils.parallel(filepaths,
    #                                         func=seisblue.io.read_hdf5_file)
    #
    #     pick_snr = seisblue.utils.parallel(instances,
    #                                        func=get_pick_snr)
    #     print(f'Get {len(pick_snr)} pick_snr.')
    #
    #     pick_snrs = [list(chain.from_iterable(pair)) for pair in zip(*pick_snr)]
    #     print(f'Get {len(pick_snrs[0])} P and {len(pick_snrs[1])} S with snr.')
    #     pick_snr_p.extend(pick_snrs[0])
    #     pick_snr_s.extend(pick_snrs[1])
    # seisblue.plot.plot_snr_distribution(pick_snr_p,
    #                                     save_dir=c['save_dir'],
    #                                     key='P',
    #                                     dataset=c['dataset'])
    # seisblue.plot.plot_snr_distribution(pick_snr_s,
    #                                     save_dir=c['save_dir'],
    #                                     key='S',
    #                                     dataset=c['dataset'])

    # seisblue.plot.create_animation(c['image_folder'], c['output_file'], fps=1)
