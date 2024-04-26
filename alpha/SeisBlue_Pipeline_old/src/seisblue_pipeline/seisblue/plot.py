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
    time_array = np.arange(instance.time_window.npts)
    time_array = time_array * instance.time_window.delta
    return time_array


def plot_dataset(instance, title=None, save_dir=None, threshold=0.5):
    if title is None:
        title = instance.trace_id

    subplot = 4
    fig = plt.figure(figsize=(8, subplot * 2))

    # plot label
    ax = fig.add_subplot(subplot, 1, subplot)

    threshold = threshold
    ax.hlines(threshold, 0, 30, lw=1, linestyles="--")
    peak_flag = []
    for i, label in enumerate(instance.labels):
        phases = [ph for ph in label.phase if ph in ('P', 'S')]
        for j, phase in enumerate(phases):
            color = _color_palette(j, i)
            ax.plot(
                _get_time_array(instance),
                label.data[:, j],
                color=color,
                label=f"{phase} {label.tag}",
            )

            peaks, _ = find_peaks(label.data[:, j], distance=100,
                                  height=threshold)

            peak_flag.append(peaks)
            ax.legend()

    # plot trace
    lines_shape = [":", "-"]
    for i, trace in enumerate(instance.traces):
        ax = fig.add_subplot(subplot, 1, i + 1)
        ax.set_ylim([-1.05, 1.05])
        if i == 0:
            plt.title(title)
        trace_data = trace.data
        ax.plot(_get_time_array(instance), trace_data, "k-", label=trace.channel)
        for j, phase in enumerate(phases):
            for k, peak in enumerate(peak_flag[j]):
                color = _color_palette(j, k)
                ax.vlines(peak_flag[j][k] / 100, -1.05, 1.05, color,
                          lines_shape[k])
        ax.legend(loc=1)

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
        plt.close()
    else:
        plt.show()
