"""
Plot
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
import numpy as np
from scipy.signal import find_peaks
import os
import seaborn as sns
import pandas as pd

import seisblue


# from cartopy.io import img_tiles
# from cartopy.mpl import ticker
# from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable
# import cartopy.crs as ccrs
# import matplotlib as mpl


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


def plot_dataset(instance, title=None, save_dir=None,
                 threshold={'P': 0.5, 'S': 0.5}, epoch_number=None):
    if title is None:
        id = '.'.join([instance.timewindow.starttime.isoformat(),
                       instance.inventory.network,
                       instance.inventory.station])
        title = id
    if epoch_number:
        savepath = f"{title}_{epoch_number}.png"
    else:
        savepath = f"{title}.png"

    subplot = 4
    fig = plt.figure(figsize=(8, subplot * 2))

    # plot label
    ax = fig.add_subplot(subplot, 1, subplot)
    time = _get_time_array(instance)

    peak_flag = []
    for i, label in enumerate(instance.labels):
        for j, phase in enumerate(label.phase[0:2]):
            color = _color_palette(j, i)
            ax.plot(
                time,
                label.data[j, :],
                color=color,
                label=f"{phase} {label.tag}",
            )

            peaks, _ = find_peaks(label.data[j, :], distance=100,
                                  height=threshold)
            if j < 2:
                peak_flag.append(peaks)

        ax.hlines(threshold, time[0], time[-1], lw=1,
                  linestyles="--")
    ax.legend()
    if len(peak_flag) == 4:
        peak_flag = [[peak_flag[0], peak_flag[1]], [peak_flag[2], peak_flag[3]]]
    elif len(peak_flag) == 2:
        peak_flag = [[peak_flag[0], peak_flag[1]], None, None]
    if ax.get_ylim()[1] < 1.5:
        ax.set_ylim([-0.05, 1.05])

    # plot trace
    lines_shape = [":", "-"]
    vline_height = [1.05, 0.5]
    stream = instance.features.data
    channels = instance.features.channel.split(',')
    for i in range(len(stream)):
        trace_data = stream[i, :]
        ax = fig.add_subplot(subplot, 1, i + 1)
        ax.set_ylim([-1.05, 1.05])
        if i == 0:
            plt.title(title)
        ax.plot(_get_time_array(instance), trace_data, "k-", label=channels[i])
        for j, label in enumerate(instance.labels):
            for k, peak in enumerate(peak_flag[j]):
                color = _color_palette(k, j)
                if peak_flag[j][k].any():
                    ax.vlines(peak_flag[j][k] / 100, -1 * vline_height[j],
                              vline_height[j], color, lines_shape[j])
        ax.legend(loc=1)

    if save_dir:
        plt.savefig(os.path.join(save_dir, savepath))
        plt.close()
    else:
        plt.show()


def plot_error_distribution(time_residuals, save_dir=None, phase=None):
    """
    Plot error distribution.

    :param time_residuals:
    :param save_dir:
    """
    bins = np.linspace(-0.5, 0.5, 100)
    plt.hist(time_residuals, bins=bins)
    plt.xticks(np.arange(-0.5, 0.51, step=0.1))
    plt.xlabel("Time residuals (sec)")
    plt.ylabel("Counts")
    plt.title(f"Error Distribution of {phase} picks")

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"error_distribution_{phase}.png"))
        plt.close()
    else:
        pass
        # plt.show()


# def plot_map(
#         geometry,
#         events,
#         center=None,
#         pad=None,
#         depth_plot=True,
#         max_depth=None,
#         depth_size=1.5,
# ):
#     """
#     Plot map.
#
#     :param geometry:
#     :param events:
#     :param boarder:
#     :param depth_plot:
#     """
#
#     stamen_terrain = img_tiles.Stamen("terrain-background")
#     fig, ax_map = plt.subplots(
#         subplot_kw={"projection": stamen_terrain.crs}, figsize=(10, 10)
#     )
#     ax_map.add_image(stamen_terrain, 8)
#
#     if events:
#         eq = []
#         for event in events:
#             eq.append([event.longitude, event.latitude, event.depth])
#         eq = sorted(eq, key=lambda eq: eq[2], reverse=True)
#         eq = np.array(eq).T
#         if max_depth is None:
#             max_depth = max(eq[2])
#         eq = stamen_terrain.crs.transform_points(
#             ccrs.PlateCarree(), eq[0], eq[1], eq[2]
#         ).T
#         ax_scatter(ax_map, eq[0], eq[1], c=eq[2], vmax=max_depth)
#
#     if geometry:
#         geom = []
#         network = geometry[0].network
#         for station in geometry:
#             geom.append([station.longitude, station.latitude])
#         geom = np.array(geom).T
#         geom = stamen_terrain.crs.transform_points(
#             ccrs.PlateCarree(), geom[0], geom[1]
#         ).T
#         ax_map.scatter(
#             geom[0],
#             geom[1],
#             label=network,
#             color="#c72c2c",
#             edgecolors="k",
#             linewidth=0.1,
#             marker="v",
#             s=50,
#         )
#
#     # PlateCarree
#     xmin, xmax = ax_map.get_xlim()
#     ymin, ymax = ax_map.get_ylim()
#
#     # convert to Google Mercator (lat lon)
#     conv = ProjectionConverter(stamen_terrain.crs, ccrs.PlateCarree())
#     xmin, ymin = conv.convert(xmin, ymin)
#     xmax, ymax = conv.convert(xmax, ymax)
#     if xmax - xmin > ymax - ymin:
#         ymin = (ymax + ymin) / 2 - (xmax - xmin) / 2
#         ymax = (ymax + ymin) / 2 + (xmax - xmin) / 2
#     else:
#         xmin = (xmax + xmin) / 2 - (ymax - ymin) / 2
#         xmax = (xmax + xmin) / 2 + (ymax - ymin) / 2
#
#     if center:
#         xmin, xmax, ymin, ymax = [
#             center[0] - pad,
#             center[0] + pad,
#             center[1] - pad,
#             center[1] + pad,
#         ]
#
#     xticks = ticker.LongitudeLocator(nbins=5)._raw_ticks(xmin, xmax)
#     yticks = ticker.LatitudeLocator(nbins=5)._raw_ticks(ymin, ymax)
#
#     # convert to PlateCarree
#     ax_map.set_xticks(xticks, crs=ccrs.PlateCarree())
#     ax_map.set_yticks(yticks, crs=ccrs.PlateCarree())
#
#     ax_map.xaxis.set_major_formatter(
#         ticker.LongitudeFormatter(zero_direction_label=True)
#     )
#     ax_map.yaxis.set_major_formatter(ticker.LatitudeFormatter())
#
#     ax_map.xaxis.set_ticks_position("both")
#     ax_map.yaxis.set_ticks_position("both")
#     ax_map.legend()
#
#     # convert PlateCarree
#     conv = ProjectionConverter(ccrs.PlateCarree(), stamen_terrain.crs)
#     xmin, ymin = conv.convert(min(xticks), min(yticks))
#     xmax, ymax = conv.convert(max(xticks), max(yticks))
#
#     ax_map.set_xlim(xmin, xmax)
#     ax_map.set_ylim(ymin, ymax)
#     divider = make_axes_locatable(ax_map)
#
#     if events and depth_plot:
#         # right depth plot
#         ax_lat_dep = divider.append_axes(
#             "right",
#             "100%",
#             pad="-25%",
#             sharey=ax_map,
#             map_projection=stamen_terrain.crs,
#         )
#         ax_lat_dep.set_aspect((1 / depth_size))
#         ax_scatter(ax_lat_dep, eq[2], eq[1], c=eq[2], vmax=max_depth)
#
#         step = max_depth / 4
#         step = np.ceil(
#             step / (10 ** np.floor(np.log10(step)))) * 10 ** np.floor(
#             np.log10(step)
#         )
#         ax_lat_dep.set_xticks([0, step * 1, step * 2, step * 3, step * 4])
#         ax_lat_dep.set_xticklabels(
#             [
#                 0,
#                 int(step / 1000),
#                 int(step * 2 / 1000),
#                 int(step * 3 / 1000),
#                 int(step * 4 / 1000),
#             ]
#         )
#         ax_lat_dep.set_xlim(0, step * 4)
#         ax_lat_dep.set_yticks(yticks, crs=ccrs.PlateCarree())
#         ax_lat_dep.yaxis.tick_right()
#         ax_lat_dep.yaxis.set_ticks_position("both")
#         ax_lat_dep.yaxis.set_tick_params(labelleft=False)
#
#         # bottom depth plot
#         ax_lon_dep = divider.append_axes(
#             "bottom",
#             "100%",
#             pad="-25%",
#             sharex=ax_map,
#             map_projection=stamen_terrain.crs,
#         )
#         ax_lon_dep.set_aspect(depth_size)
#         ax_lon_dep.set_xlim(xmin, xmax)
#         ax_scatter(ax_lon_dep, eq[0], eq[2], c=eq[2], vmax=max_depth)
#
#         ax_lon_dep.set_yticks([0, step * 1, step * 2, step * 3, step * 4])
#         ax_lon_dep.set_yticklabels(
#             [
#                 0,
#                 int(step / 1000),
#                 int(step * 2 / 1000),
#                 int(step * 3 / 1000),
#                 int(step * 4 / 1000),
#             ]
#         )
#         ax_lon_dep.set_ylim(0, step * 4)
#         ax_lon_dep.xaxis.set_ticks_position("both")
#         ax_lon_dep.set_xticks(xticks, crs=ccrs.PlateCarree())
#         ax_lon_dep.invert_yaxis()
#     plt.tight_layout()
#     plt.show()
#
# def ax_scatter(
#         ax,
#         x_data,
#         y_data,
#         c=None,
#         vmax=None,
#         event_depth_cmap=mpl.cm.turbo_r,
#         event_mark_size=10,
#         alpha=0.7,
# ):
#     axes = ax.scatter(
#         x_data,
#         y_data,
#         label="Event",
#         c=c,
#         cmap=event_depth_cmap,
#         vmin=0,
#         vmax=vmax,
#         edgecolors="k",
#         linewidth=0.1,
#         marker="o",
#         alpha=alpha,
#         s=event_mark_size,
#     )
#     return axes


def plot_metrics_by_epoch(metrics, epochs, figdir):
    for i, (key, value) in enumerate(metrics.items()):
        plt.plot(epochs, value, label=key, marker='o', linestyle='-',)
    plt.xticks(range(10, 100, 10))
    plt.title('metrics_per_epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.savefig(f'{figdir}/metrics_per_epoch.png')


def plot_metrics_by_threshold(metrics, thresholds, figdir, old_metrics=None,
                              old_metrics_threshold=None):
    plt.figure(figsize=(10, 8))
    color = ['firebrick', 'darkorange']
    for i, (key, value) in enumerate(metrics.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(thresholds, value, label='delta=0.2s', marker='o',
                 linestyle='-', color=color[0])
        for i, j in zip(thresholds, value):
            plt.text(i, j, str(round(j, 2)), ha='center', va='bottom',
                     color=color[0], fontsize=8)

        if old_metrics[key]:
            plt.plot(old_metrics_threshold, old_metrics[key], label='delta=0.1s',
                     marker='o', linestyle='-', color=color[1])
            for i, j in zip(old_metrics_threshold, old_metrics[key]):
                plt.text(i, j, str(round(j, 2)), ha='center', va='bottom',
                         color=color[1], fontsize=8)
            plt.title(key)
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figdir}/metrics_by_threshold_delta.png')


def plot_snr_distribution(pick_snr, save_dir=None, key=None, dataset=None):
    """
    Plot signal to noise ratio distribution.

    :param pick_snr:
    :param save_dir:
    """
    sns.set()
    bins = np.linspace(-1, 10, 55)
    plt.hist(pick_snr, bins=bins)
    plt.semilogy()
    plt.xticks(np.arange(-1, 11, step=1))
    plt.xlabel("Signal to Noise Ratio (log10)")
    plt.ylabel("Counts")
    title = f"SNR Distribution of {key}" if key and dataset else "SNR Distribution"
    plt.title(title)

    if save_dir:
        seisblue.utils.make_dirs(save_dir)
        savename = f"{dataset}_snr_distribution_{key}.png" if key else "snr_distribution.png"
        plt.savefig(os.path.join(save_dir, savename))
        plt.close()
    else:
        plt.show()


def create_animation(image_folder, output_file, fps=2):
    """
    Creates an animation from a sequence of images.

    :param str image_folder: Path to the folder containing image files.
    :param str output_file: Path and filename of the output animation file.
    :param int fps: Frames per second in the animation.
    """
    images = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith((".png", ".jpg"))]
    fig, ax = plt.subplots()

    def update(frame):
        img = mpimg.imread(images[frame])
        ax.clear()
        ax.imshow(img)
        ax.axis('off')

    ani = FuncAnimation(fig, update, frames=len(images), repeat=False)
    ani.save(output_file, fps=fps, writer='ffmpeg')

    plt.close()
