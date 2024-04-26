# -*- coding: utf-8 -*-
"""
Plot
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import os


def _color_palette(color=1, shade=1):
    """
    Return a color palette form a selected color and shade level.

    :param int color: (Optional.) 0=Blue, 1=Deep Orange, 2=Green, default is 1.
    :param int shade: (Optional.) 0=light, 1=regular, 2=dark, default is 1.
    :rtype: str
    :return: Hex color code.
    """
    palette = [
        ["#90CAF9", "#2196F3", "#1565C0"],  # Blue
        ["#FFAB91", "#FF5722", "#D84315"],  # Deep Orange
        ["#A5D6A7", "#4CAF50", "#2E7D32"],  # Green
    ]

    return palette[color][shade]


def _get_time_array(instance):
    """
    Returns time step array from feature dict.

    :param instance: Data instance.
    :rtype: numpy.array
    :return: Time array.
    """
    time_array = np.arange(instance.timewindow.npts)
    time_array = time_array * instance.timewindow.delta
    return time_array


def plot_dataset(instance, title=None, save_dir=None, threshold=0.5):
    labels = [f'{label.tag}:{label.pick.polarity}' for label in instance.labels]
    peak = 156 / instance.timewindow.samplingrate
    diff_X = _get_time_array(instance)[:-1] - (instance.timewindow.delta / 2)
    polarity_confidence = instance.labels.pick.confidence
    if title is None:
        title = f"{instance.id} score: ({polarity_confidence})"

    subplot = 6
    fig = plt.figure(figsize=(8, subplot * 2))
    plt.tilte(title)
    for i, trace in enumerate(instance.traces):
        # plot trace
        ax = fig.add_subplot(subplot, 1, i + 1)
        ax.set_ylim([-1.05, 1.05])
        if i == 0:
            plt.title(title + str(labels))
        ax.plot(_get_time_array(instance), trace.data, "k-",
                label=trace.channel)
        ax.vlines(peak, -1.05, 1.05, 'k', '--')
        ax.legend(loc=1)

        # plot signed diff
        diff_Y = np.sign(np.diff(trace.data))
        diff_Y[:156] = 0
        ax = fig.add_subplot(subplot, 1, i + 4)
        ax.plot(diff_X, diff_Y, "b-", label='diff_' + trace.channel)

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
        plt.close()
    else:
        plt.show()


def plot_polarity(instance):
    peak = 156 / instance.timewindow.samplingrate
    trace = [trace for trace in instance.traces if list(trace.channel)[-1] == 'Z']

    # plot trace
    plt.figure()
    plt.ylim([-1.05, 1.05])
    plt.title(instance.id)
    plt.plot(_get_time_array(instance), trace.data, "k-",
            label=trace.channel)
    plt.axvline(peak, 'k', '--')
    plt.title(f'{instance.inventory.network}.{instance.inventory.station}: {instance.event.time} {instance.event.magnitude}')
    plt.text(0, 1, '')
    plt.legend(loc=1)






