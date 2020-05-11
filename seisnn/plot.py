"""
Plot
=============

.. autosummary::
    :toctree: plot

    color_palette
    get_time_array
    plot_confusion_matrix
    plot_dataset
    plot_error_distribution
    plot_loss
    plot_snr_distribution

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LongitudeLocator, LatitudeLocator

import seaborn as sns
from obspy import UTCDateTime
from seisnn.utils import make_dirs
from seisnn.qc import signal_to_noise_ratio, precision_recall_f1_score


def color_palette(color=1, shade=1):
    # color palette source:
    # http://www.webfreelancer.com.br/color/colors.html

    # shade:     #200       #500       #800
    palette = [['#90CAF9', '#2196F3', '#1565C0'],  # Blue
               ['#FFAB91', '#FF5722', '#D84315'],  # Deep Orange
               ['#A5D6A7', '#4CAF50', '#2E7D32']]  # Green

    return palette[color][shade]


def get_time_array(feature):
    time_array = np.arange(feature['npts'])
    time_array = time_array * feature['delta']
    return time_array


def plot_dataset(feature, snr=False, enlarge=False, xlim=None, title=None, save_dir=None):
    if title is None:
        title = f'{feature["starttime"]}_{feature["id"][:-3]}'
    # if feature['pick_time']:
    #     first_pick_time = UTCDateTime(feature['pick_time'][-1]) - UTCDateTime(feature['starttime'])
    # else:
    #     first_pick_time = 1

    subplot = len(feature['channel']) + 1
    fig = plt.figure(figsize=(8, subplot * 2))
    for i, chan in enumerate(feature['channel']):
        ax = fig.add_subplot(subplot, 1, i + 1)
        plt.title(title + chan)

        # if xlim:
        #     plt.xlim(xlim)
        # if enlarge:
        #     plt.xlim((first_pick_time - 1, first_pick_time + 2))
        trace = feature['trace'][-1, :, i]
        ax.plot(get_time_array(feature), trace, "k-", label=chan)
        y_min, y_max = ax.get_ylim()

        # if feature['pick_time']:
        #     label_set = set()
        #     pick_type = ['manual', 'predict']
        #
        #     for i in range(len(feature['pick_time'])):
        #         pick_set = feature['pick_set'][i]
        #         pick_phase = feature['pick_phase'][i]
        #         phase_color = feature['phase'].index(pick_phase)
        #         type_color = pick_type.index(pick_set)
        #
        #         color = color_palette(type_color, 1)
        #         label = pick_set + " " + pick_phase
        #
        #         pick_time = UTCDateTime(feature['pick_time'][i]) - UTCDateTime(feature['starttime'])
        #         if not label in label_set:
        #             ax.vlines(pick_time, y_min, y_max, color=color, lw=1,
        #                       label=label)
        #             label_set.add(label)
        #         else:
        #             ax.vlines(pick_time, y_min, y_max, color=color, lw=1)
        #
        #         if snr and pick_set == 'manual':
        #             try:
        #                 index = int(pick_time / feature['delta'])
        #                 noise = trace[index - 100:index]
        #                 signal = trace[index: index + 100]
        #                 snr = signal_to_noise_ratio(signal, noise)
        #                 if not snr == float("inf"):
        #                     ax.text(pick_time, y_max-0.1, f'SNR: {snr:.2f}')
        #             except IndexError:
        #                 pass
        ax.legend(loc=1)

    ax = fig.add_subplot(subplot, 1, subplot)
    ax.set_ylim([-0.05, 1.05])

    for i in range(feature['pdf'].shape[2]):
        if feature['phase'][i]:
            color = color_palette(i, 1)
            ax.plot(get_time_array(feature), feature['pdf'][-1, :, i], color=color, label=feature['phase'][i])
            ax.legend()

        else:
            label_only = [Line2D([0], [0], color="#AAAAAA", lw=2)]
            ax.legend(label_only, ['No phase data'])

    threshold = 0.5
    ax.hlines(threshold, 0, 30, lw=1, linestyles='--')

    # if xlim:
    #     plt.xlim(xlim)
    # if enlarge:
    #     plt.xlim((first_pick_time - 1, first_pick_time + 2))

    if save_dir:
        make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title}.png'))
        plt.close()
    else:
        plt.show()


def plot_loss(log_file, save_dir=None):
    loss = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            loss.append(line)

    file_name = os.path.basename(log_file).split('.')
    loss = np.asarray(loss).astype(np.float32)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(loss[:, 0], label='train')
    ax.plot(loss[:, 1], label='validation')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    ax.legend()
    plt.title(f'{file_name[0]} loss')

    if save_dir:
        make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{file_name[0]}.png'))
        plt.close()
    else:
        plt.show()


def plot_error_distribution(time_residuals, save_dir=None):
    bins = np.linspace(-0.5, 0.5, 100)
    plt.hist(time_residuals, bins=bins)
    plt.xticks(np.arange(-0.5, 0.51, step=0.1))
    plt.xlabel("Time residuals (sec)")
    plt.ylabel("Counts")
    plt.title("Error Distribution")

    if save_dir:
        make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'error_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_snr_distribution(pick_snr, save_dir=None):
    sns.set()
    bins = np.linspace(-1, 10, 55)
    plt.hist(pick_snr, bins=bins)
    plt.xticks(np.arange(-1, 11, step=1))
    plt.xlabel("Signal to Noise Ratio (log10)")
    plt.ylabel("Counts")
    plt.title("SNR Distribution")

    if save_dir:
        make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'error_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(true_positive, pred_count, val_count):
    matrix = pd.DataFrame([[true_positive, pred_count - true_positive],
                           [val_count - true_positive, 0]], columns=['True', 'False'], index=['True', 'False'])

    precision, recall, f1 = precision_recall_f1_score(true_positive, pred_count, val_count)

    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, cbar=False, fmt="d", cmap='Blues', square=True)
    bottom, top = plt.ylim()
    plt.ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title(f'Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}')
    plt.show()
    sns.set(font_scale=1)


def plot_map(geometry, events):
    stamen_terrain = Stamen('terrain-background')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
    ax.add_image(stamen_terrain, 11)

    if events:
        eq = []
        for event in events:
            eq.append([event.longitude, event.latitude])
        eq = np.array(eq).T
        ax.scatter(eq[0], eq[1], label='Event', transform=ccrs.PlateCarree(),
                   color='#555555', edgecolors='k', linewidth=0.3, marker='o', s=10)

    if geometry:
        geom = []
        network = geometry[0].network
        for station in geometry:
            geom.append([station.longitude, station.latitude])
        geom = np.array(geom).T
        ax.scatter(geom[0], geom[1], label=network, transform=ccrs.PlateCarree(),
                   color='#c72c2c', edgecolors='k', linewidth=0.1, marker='v', s=40)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    min_point = max_point = TransferProjection(stamen_terrain.crs, ccrs.PlateCarree())
    xmin, ymin = min_point.transfer(xmin, ymin)
    xmax, ymax = max_point.transfer(xmax, ymax)

    xticks = LongitudeLocator(nbins=2)._raw_ticks(xmin, xmax)
    yticks = LatitudeLocator(nbins=3)._raw_ticks(ymin, ymax)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.legend(markerscale=1)
    plt.show()


class TransferProjection:
    x = y = None

    def __init__(self, source_proj, target_proj):
        self.source_proj = source_proj
        self.target_proj = target_proj

    def transfer(self, x, y):
        self.x = x
        self.y = y
        result = self.target_proj._project_point(self, self.source_proj)
        return result.x, result.y


if __name__ == "__main__":
    pass
