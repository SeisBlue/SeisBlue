import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from seisnn.pick import get_picks_from_dataset
from seisnn.utils import make_dirs


def color_palette(feature, phase=None, shade=1):
    # color palette source:
    # http://www.webfreelancer.com.br/color/colors.html

    # shade:     #200       #500       #800
    palette = [['#90CAF9', '#2196F3', '#1565C0'],  # Blue
               ['#FFAB91', '#FF5722', '#D84315'],  # Deep Orange
               ['#A5D6A7', '#4CAF50', '#2E7D32'],  # Green
               ]

    phase_list = list(feature['phase'].keys())
    phase_index = phase_list.index(phase)
    return palette[phase_index][shade]


def get_time_array(feature):
    time_array = np.arange(feature['npts'])
    time_array = time_array * feature['delta']
    return time_array


def plot_dataset(feature, enlarge=False, xlim=None, title=None, save_dir=None):
    start_time = feature['starttime']
    time_stamp = feature['starttime'].isoformat()
    picks = feature['picks']

    if title is None:
        title = f'{time_stamp}_{feature["id"][:-3]}'
    if not picks.empty:
        first_pick_time = picks['pick_time'].values[0] - start_time
    else:
        first_pick_time = 1

    subplot = len(feature['channel']) + 1

    fig = plt.figure(figsize=(8, subplot * 2))
    for i, chan in enumerate(feature['channel']):
        ax = fig.add_subplot(subplot, 1, i + 1)

        plt.title(title + chan)

        if xlim:
            plt.xlim(xlim)
        if enlarge:
            plt.xlim((first_pick_time - 1, first_pick_time + 2))
        ax.plot(get_time_array(feature), feature['channel'][chan], "k-", label=chan)
        y_min, y_max = ax.get_ylim()

        if not picks.empty:
            pick_set_list = picks['pick_set'].unique().tolist()
            labelset = set()
            for i in range(len(picks)):
                pick_set = picks['pick_set'].values[i]
                pick_phase = picks['pick_phase'].values[i]
                pick_index = pick_set_list.index(pick_set)

                color = color_palette(feature, pick_phase, pick_index)
                label = pick_set + " " + pick_phase

                pick_time = picks['pick_time'].values[i] - start_time
                if not label in labelset:
                    ax.vlines(pick_time, y_min / (pick_index + 1), y_max / (pick_index + 1), color=color, lw=2,
                              label=label)
                    labelset.add(label)
                else:
                    ax.vlines(pick_time, y_min / (pick_index + 1), y_max / (pick_index + 1), color=color, lw=2)

        ax.legend(loc=1)

    ax = fig.add_subplot(subplot, 1, subplot)

    if feature['phase']:
        for phase in feature['phase']:
            color = color_palette(feature, phase)
            ax.plot(get_time_array(feature), feature['phase'][phase], color=color, label=phase)
        ax.legend()

    else:
        label_only = [Line2D([0], [0], color="#AAAAAA", lw=2)]
        ax.legend(label_only, ['No phase data'])

    if xlim:
        plt.xlim(xlim)
    if enlarge:
        plt.xlim((first_pick_time - 1, first_pick_time + 2))

    if save_dir:
        make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title}.png'))
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
