"""
Plot
"""

import os
from subprocess import call, DEVNULL, STDOUT

from cartopy.io import img_tiles
from cartopy.mpl import ticker
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory
from obspy import Stream, UTCDateTime
from obspy.clients.filesystem import sds
from obspy.geodetics import locations2degrees
from scipy.signal import find_peaks
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

import seisblue
import seisblue.utils
import seisblue.qc


def color_palette(color=1, shade=1):
    """
    Return a color palette form a selected color and shade level.

    :param int color: (Optional.) 0=Blue, 1=Deep Orange, 2=Green, default is 1.
    :param int shade: (Optional.) 0=light, 1=regular, 2=dark, default is 1.
    :rtype: str
    :return: Hex color code.
    """
    palette = [
        ['#90CAF9', '#2196F3', '#1565C0'],  # Blue
        ['#FFAB91', '#FF5722', '#D84315'],  # Deep Orange
        ['#A5D6A7', '#4CAF50', '#2E7D32'],  # Green
    ]

    return palette[color][shade]


def get_time_array(instance):
    """
    Returns time step array from feature dict.

    :param instance: Data instance.
    :rtype: numpy.array
    :return: Time array.
    """
    time_array = np.arange(instance.metadata.npts)
    time_array = time_array * instance.metadata.delta
    return time_array


def plot_dataset(instance, title=None, save_dir=None, threshold=0.5):
    """
    Plot trace and label.

    :param instance:
    :param title:
    :param save_dir:
    :param threshold:
    """
    if title is None:
        title = f'{instance.metadata.starttime}_{instance.metadata.id[:-3]}'

    subplot = 4
    fig = plt.figure(figsize=(8, subplot * 2))

    # plot label
    ax = fig.add_subplot(subplot, 1, subplot)

    threshold = threshold
    ax.hlines(threshold, 0, 30, lw=1, linestyles='--')
    peak_flag = []
    for i, label in enumerate([instance.label, instance.predict]):
        for j, phase in enumerate(label.phase[0:2]):
            color = color_palette(j, i)
            ax.plot(get_time_array(instance),
                    label.data[-1, :, j],
                    color=color, label=f'{phase} {label.tag}')

            peaks, _ = find_peaks(label.data[-1, :, j],
                                  distance=100,
                                  height=threshold)
            if j < 2:
                peak_flag.append(peaks)
            ax.legend()

    peak_flag = [[peak_flag[0], peak_flag[1]], [peak_flag[2], peak_flag[3]]]
    if ax.get_ylim()[1] < 1.5:
        ax.set_ylim([-0.05, 1.05])

    # plot trace
    lines_shape = [':', '-']
    for i, chan in enumerate(instance.trace.channel):
        ax = fig.add_subplot(subplot, 1, i + 1)
        ax.set_ylim([-1.05, 1.05])
        if i == 0:
            plt.title(title[0:-2])
        trace = instance.trace.data[-1, :, i]
        ax.plot(get_time_array(instance), trace, "k-", label=chan)
        for j, phase in enumerate(['label', 'predict']):
            for k, peak in enumerate(peak_flag[j]):
                color = color_palette(k, j)
                ax.vlines(peak_flag[j][k] / 100, -1.05, 1.05, color,
                          lines_shape[j])
        ax.legend(loc=1)

    if save_dir:
        seisblue.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title}.png'))
        plt.close()
    else:
        plt.show()


def plot_loss(log_file, save_dir=None):
    """
    Plot loss history.

    :param log_file:
    :param save_dir:
    """
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
        seisblue.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{file_name[0]}.png'))
        plt.close()
    else:
        plt.show()


def plot_error_distribution(time_residuals, save_dir=None):
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
    plt.title("Error Distribution")

    if save_dir:
        seisblue.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'error_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_PGA_distribution(pga_list, save_dir=None):
    sns.set()
    bins = np.linspace(0, 1000, 100)
    # bins = [0.5,1,2,3,4,5,6,7,8.0,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,23,24,25]
    plt.hist(pga_list, bins=bins, )

    plt.xticks(np.arange(0, 1001, step=100))
    plt.semilogy()
    plt.xlabel("PGA(gal)")
    plt.ylabel("Counts")
    plt.title("PGA distribution")
    plt.show()


def plot_event_magnitudes_distribution(events, save_dir=None):
    """
        plot_event_magnitude_distribution.

        :param magnitude_list:
        :param save_dir:
        """
    magnitudes_list = []
    for event in events:
        magnitudes_list.append(event.magnitudes)
    sns.set()
    bins = np.linspace(-2, 9, 110)
    plt.hist(magnitudes_list, bins=bins)
    plt.xticks(np.arange(-2, 9, step=1))
    plt.semilogy()
    plt.xlabel("Magnitude")
    plt.ylabel("Counts")
    plt.title("Magnitudes distribution")

    if save_dir:
        seisblue.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'error_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_snr_distribution(pick_snr, save_dir=None):
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
    plt.title("SNR Distribution")

    if save_dir:
        seisblue.utils.make_dirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'error_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(true_positive, pred_count, val_count):
    """
    Plot confusion matrix.

    :param true_positive:
    :param pred_count:
    :param val_count:
    """
    matrix = pd.DataFrame([[true_positive, pred_count - true_positive],
                           [val_count - true_positive, 0]],
                          columns=['True', 'False'],
                          index=['True', 'False'])

    precision, recall, f1 = seisblue.qc.precision_recall_f1_score(true_positive,
                                                                  pred_count,
                                                                  val_count)

    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, cbar=False, fmt="d", cmap='Blues',
                square=True)
    bottom, top = plt.ylim()
    plt.ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title(
        f'Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}')
    plt.show()
    sns.set(font_scale=1)


def ax_scatter(ax, x_data, y_data, c=None, vmax=None,
               event_depth_cmap=mpl.cm.GnBu,
               event_mark_size=10):
    ax.scatter(x_data, y_data,
               label='Event',
               c=c,
               cmap=event_depth_cmap,
               vmin=0,
               vmax=vmax,
               edgecolors='k',
               linewidth=0.1,
               marker='o',
               alpha=0.7,
               s=event_mark_size)


def plot_map(geometry, events, center=None, pad=None, depth_plot=True,
             max_depth=None, depth_size=1.5):
    """
    Plot map.

    :param geometry:
    :param events:
    :param boarder:
    :param depth_plot:
    """

    stamen_terrain = img_tiles.Stamen('terrain-background')
    fig, ax_map = plt.subplots(
        subplot_kw={'projection': stamen_terrain.crs},
        figsize=(10, 10)
    )
    ax_map.add_image(stamen_terrain, 8)

    if events:
        eq = []
        for event in events:
            eq.append([event.longitude, event.latitude, event.depth])
        eq = sorted(eq, key=lambda eq: eq[2], reverse=True)
        eq = np.array(eq).T
        if max_depth is None:
            max_depth = max(eq[2])
        eq = stamen_terrain.crs.transform_points(ccrs.PlateCarree(),
                                                 eq[0], eq[1], eq[2]).T
        ax_scatter(ax_map, eq[0], eq[1], c=eq[2], vmax=max_depth)

    if geometry:
        geom = []
        network = geometry[0].network
        for station in geometry:
            geom.append([station.longitude, station.latitude])
        geom = np.array(geom).T
        geom = stamen_terrain.crs.transform_points(ccrs.PlateCarree(),
                                                   geom[0], geom[1]).T
        ax_map.scatter(geom[0], geom[1],
                       label=network,
                       color='#c72c2c',
                       edgecolors='k',
                       linewidth=0.1,
                       marker='v',
                       s=50)

    # PlateCarree
    xmin, xmax = ax_map.get_xlim()
    ymin, ymax = ax_map.get_ylim()

    # convert to Google Mercator (lat lon)
    conv = ProjectionConverter(stamen_terrain.crs, ccrs.PlateCarree())
    xmin, ymin = conv.convert(xmin, ymin)
    xmax, ymax = conv.convert(xmax, ymax)
    if xmax - xmin > ymax - ymin:
        ymin = (ymax + ymin) / 2 - (xmax - xmin) / 2
        ymax = (ymax + ymin) / 2 + (xmax - xmin) / 2
    else:
        xmin = (xmax + xmin) / 2 - (ymax - ymin) / 2
        xmax = (xmax + xmin) / 2 + (ymax - ymin) / 2

    if center:
        xmin, xmax, ymin, ymax = [center[0] - pad, center[0] + pad,
                                  center[1] - pad, center[1] + pad]

    xticks = ticker.LongitudeLocator(nbins=5)._raw_ticks(xmin, xmax)
    yticks = ticker.LatitudeLocator(nbins=5)._raw_ticks(ymin, ymax)

    # convert to PlateCarree
    ax_map.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax_map.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax_map.xaxis.set_major_formatter(
        ticker.LongitudeFormatter(zero_direction_label=True))
    ax_map.yaxis.set_major_formatter(ticker.LatitudeFormatter())

    ax_map.xaxis.set_ticks_position('both')
    ax_map.yaxis.set_ticks_position('both')
    ax_map.legend()

    # convert PlateCarree
    conv = ProjectionConverter(ccrs.PlateCarree(), stamen_terrain.crs)
    xmin, ymin = conv.convert(min(xticks), min(yticks))
    xmax, ymax = conv.convert(max(xticks), max(yticks))

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    divider = make_axes_locatable(ax_map)

    if events and depth_plot:
        # right depth plot
        ax_lat_dep = divider.append_axes("right", '100%', pad='-25%',
                                         sharey=ax_map,
                                         map_projection=stamen_terrain.crs)
        ax_lat_dep.set_aspect((1 / depth_size))
        ax_scatter(ax_lat_dep, eq[2], eq[1], c=eq[2], vmax=max_depth)

        step = max_depth / 4
        step = np.ceil(
            step / (10 ** np.floor(np.log10(step)))) * 10 ** np.floor(
            np.log10(step))
        ax_lat_dep.set_xticks([0, step * 1, step * 2, step * 3, step * 4])
        ax_lat_dep.set_xticklabels([0,
                                    int(step / 1000),
                                    int(step * 2 / 1000),
                                    int(step * 3 / 1000),
                                    int(step * 4 / 1000)])
        ax_lat_dep.set_xlim(0, step * 4)
        ax_lat_dep.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax_lat_dep.yaxis.tick_right()
        ax_lat_dep.yaxis.set_ticks_position('both')
        ax_lat_dep.yaxis.set_tick_params(labelleft=False)

        # bottom depth plot
        ax_lon_dep = divider.append_axes("bottom", '100%', pad='-25%',
                                         sharex=ax_map,
                                         map_projection=stamen_terrain.crs)
        ax_lon_dep.set_aspect(depth_size)
        ax_lon_dep.set_xlim(xmin, xmax)
        ax_scatter(ax_lon_dep, eq[0], eq[2], c=eq[2], vmax=max_depth)

        ax_lon_dep.set_yticks([0, step * 1, step * 2, step * 3, step * 4])
        ax_lon_dep.set_yticklabels([0,
                                    int(step / 1000),
                                    int(step * 2 / 1000),
                                    int(step * 3 / 1000),
                                    int(step * 4 / 1000)])
        ax_lon_dep.set_ylim(0, step * 4)
        ax_lon_dep.xaxis.set_ticks_position('both')
        ax_lon_dep.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax_lon_dep.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_event_trace(db, event,station_code):
    stream_plot = Stream()
    plot_station = []
    Ptime=[]
    tfr_converter = seisblue.components.TFRecordConverter(trace_length =60)
    for pick in event.picks:
        arr = []
        if pick.waveform_id.station_code not in plot_station:
            plot_station.append(pick.waveform_id.station_code)
            metadata = tfr_converter.get_time_window(
                anchor_time=event.origins[0].time - 5,
                station=pick.waveform_id.station_code,
                shift=0)
            inventory = db.get_inventories(
                station=pick.waveform_id.station_code)
            fmtstr = os.path.join(
                "{year}", "{doy:03d}", "{station}",
                "{station}.{network}.{location}.{channel}.{year}.{doy:03d}")
            stream = seisblue.io.read_sds(metadata, channel='*Z')

            arr.append(pick.time)

            for item in stream.items():
                tmp_trace = item[1].traces[0]

                tmp_trace.resample(100)
                tmp_trace.filter('bandpass', freqmin=1, freqmax=45)
                tmp_trace.detrend('linear')
                tmp_trace.detrend('demean')
                tmp_trace.normalize()
                tmp_trace.trim(event.origins[0].time)
                tmp_trace.stats["coordinates"] = {}
                tmp_trace.stats["coordinates"]["latitude"] = inventory[
                    0].latitude
                tmp_trace.stats["coordinates"]["longitude"] = inventory[
                    0].longitude
                stream_plot.append(item[1].traces[0])
                Ptime.append(pick.time)
    fig = plt.figure()
    stream_plot.plot(type='section',
                     dist_degree=True,
                     scale=0.3,
                     time_down=True,
                     ev_coord=(event.origins[0].latitude,
                               event.origins[0].longitude),
                     offset_min=0,
                     show=False, fig=fig)
    ax = fig.axes[0]
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for i,tr in enumerate(stream_plot):
        offset = locations2degrees(
            tr.stats.coordinates.latitude,
            tr.stats.coordinates.longitude,
            event.origins[0].latitude, event.origins[0].longitude)
        ax.text(offset, 1.0, tr.stats.station, rotation=270,
                va="bottom", ha="center", transform=transform, zorder=10)
        ax.plot(offset,Ptime[i],'ro')
    plt.subplots_adjust(top=0.85)
    ax.set_title(str(event.origins[0].time) + '\n\n', loc='center')

    plt.show()
    call(f'rm /home/andy/Catalog/plot_sfile_dir/*',
         shell=True,
         stdout=DEVNULL,
         stderr=STDOUT)


class ProjectionConverter:
    """
    Cartopy projection converter.
    """

    def __init__(self, source_proj, target_proj):
        self.x = None
        self.y = None
        self.source_proj = source_proj
        self.target_proj = target_proj

    def convert(self, x, y):
        """
        Returns converted project location.

        :param float x: X location.
        :param float y: Y location.
        :rtype: float
        :return: (x, y)
        """
        self.x = x
        self.y = y
        result = self.target_proj._project_point(self, self.source_proj)
        return result.x, result.y


if __name__ == "__main__":
    pass


def plot_pick_stream(pick, fmtstr=None):
    config = seisblue.utils.Config()

    client = sds.Client(sds_root=config.sds_root)
    if fmtstr:
        client.FMTSTR = fmtstr
    stream = client.get_waveforms(network="*",
                                  station=pick.station,
                                  location="*",
                                  channel='*',
                                  starttime=UTCDateTime(pick.time) - 10,
                                  endtime=UTCDateTime(pick.time) + 20
                                  )
    if stream:
        stream.resample(100)
        stream.detrend('linear')
        stream.detrend('demean')
        stream.filter('bandpass', freqmin=1, freqmax=45)
        stream.normalize()
        stream.plot()


def plot_event_residual(tp_event,tp_origin_event):
    erlt = []
    erln = []
    erdp = []

    for i in range(len(tp_event)):
        erlt.append(tp_event[i].latitude * 110 - tp_origin_event[i].latitude * 110)
        erln.append(tp_event[i].longitude * 110 - tp_origin_event[i].longitude* 110)
        erdp.append(tp_event[i].depth / 1000 - tp_origin_event[i].depth/1000)
    for i in [erlt, erln, erdp]:
        sns.set()
        sns.histplot(i,bins=np.linspace(-14.5,15.5,60))
        # plt.semilogy()
        plt.xticks(np.arange(np.round(min(i), decimals=-1), np.round(max(i), decimals=-1), step=1), rotation='vertical')
        plt.ylabel("Counts")
        plt.xlim(-15,15)
        plt.tight_layout()
        plt.show()

    data_dict = {'longitude': erln,
                 'latitude': erlt,
                 'depth': erdp}

    pairs = [['longitude', 'latitude'],
             ['longitude', 'depth'],
             ['depth', 'latitude']]

    for x, y in pairs:
        sns.kdeplot(data_dict[x], data_dict[y], shade=True, color="m")
        sns.scatterplot(data_dict[x], data_dict[y], marker="+", linewidth=0.3, color="k")
        plt.title(f'error_distribution ({x}-{y})')
        plt.xlabel(f'{x} error (km)')
        plt.ylabel(f'{y} error (km)')
        plt.xlim(-15,15)
        plt.ylim(-15,15)
        plt.show()


    for i in ['longitude', 'latitude', 'depth']:
        print(f'{i} standard deviation = {np.std(data_dict[i])}')
        print(f'{i} median = {np.median(data_dict[i])}')
        print(f'{i} mean = {np.mean(data_dict[i])}')