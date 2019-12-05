import os

import matplotlib.pyplot as plt
import numpy as np

from seisnn.pick import get_picks_from_dataset


def color_palette(feature, phase=None, shade=1):
    # color palette source: http://www.webfreelancer.com.br/color/colors.html
    # 200       #500       #800
    palette = [['#90CAF9', '#2196F3', '#1565C0'],  # Blue
               ['#A5D6A7', '#4CAF50', '#2E7D32'],  # Green
               ['#FFAB91', '#FF5722', '#D84315']]  # Deep Orange
    phase_index = feature['phase'].index(phase)
    return palette[phase_index][shade]


def pick_exist(feature):
    if not feature['pick_phase'][0] == 'NA':
        return True
    else:
        return False


def get_time_array(feature):
    time_array = np.arange(feature['npts'])
    time_array = time_array * feature['delta']
    return time_array


def plot_dataset(feature, enlarge=False, xlim=None, save_dir=None):
    start_time = feature['starttime']
    time_stamp = feature['starttime'].isoformat()

    if pick_exist(feature):
        first_pick_time = feature['pick_time'][0] - start_time
    else:
        first_pick_time = 1

    subplot = len(feature['channel']) + 1

    fig = plt.figure(figsize=(8, subplot * 2))
    for i, chan in enumerate(feature['channel']):
        ax = fig.add_subplot(subplot, 1, i + 1)
        plt.title(time_stamp + " " + feature['id'][:-3] + chan)

        if xlim:
            plt.xlim(xlim)
        if enlarge:
            plt.xlim((first_pick_time - 1, first_pick_time + 2))
        ax.plot(get_time_array(feature), feature[chan], "k-", label=chan)
        y_min, y_max = ax.get_ylim()

        if pick_exist(feature):
            pick_set_list = list(set(feature['pick_set']))
            labelset = set()
            for i in range(len(feature['pick_time'])):
                pick_set = feature['pick_set'][i]
                pick_phase = feature['pick_phase'][i]
                pick_set_index = pick_set_list.index(pick_set)

                color = color_palette(feature, pick_phase, pick_set_index)
                label = pick_set + " " + pick_phase

                pick_time = feature['pick_time'][i] - start_time
                if not label in labelset:
                    ax.vlines(pick_time, y_min/(pick_set_index+1), y_max/(pick_set_index+1), color=color, lw=2, label=label)
                    labelset.add(label)
                else:
                    ax.vlines(pick_time, y_min/(pick_set_index+1), y_max/(pick_set_index+1), color=color, lw=2)

        ax.legend(loc=1)

    ax = fig.add_subplot(subplot, 1, subplot)
    for phase in feature['phase']:
        color = color_palette(feature, phase)
        ax.plot(get_time_array(feature), feature[phase], color=color, label=phase)
    if xlim:
        plt.xlim(xlim)
    if enlarge:
        plt.xlim((first_pick_time - 1, first_pick_time + 2))
    ax.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "/" + time_stamp + "_" + feature['id'])
        plt.close()
    else:
        plt.show()


def plot_error_distribution(predict_pkl_list):
    time_residuals = []
    for i, pkl in enumerate(predict_pkl_list):
        picks = get_picks_from_dataset(pkl)
        for p in picks:
            if p.time_errors:
                residual = p.time_errors.get("uncertainty")
                time_residuals.append(float(residual))

        if i % 1000 == 0:
            print("Reading... %d out of %d " % (i, len(predict_pkl_list)))

    bins = np.linspace(-0.5, 0.5, 100)
    plt.hist(time_residuals, bins=bins)
    plt.xticks(np.arange(-0.5, 0.51, step=0.1))
    plt.xlabel("Time residuals (sec)")
    plt.ylabel("Number of picks")
    plt.title("Error Distribution")
    plt.show()
