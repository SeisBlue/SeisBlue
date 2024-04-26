# -*- coding: utf-8 -*-
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from obspy.io.nordic.core import _write_nordic

from seisblue import core, tool, io


def analysis_distribution_of_stations(events_dir, filename, thres, out_dir, network, out_file=None):
    events_path_list = glob.glob(os.path.join(events_dir, filename))[0]
    obspy_events = io.get_obspy_event(events_path_list)
    print(f'Read {len(obspy_events)} events.')
    stations_count = []
    for event in obspy_events:
        stations = []
        ARC = []
        ot = event.origins[0].time
        for pick in event.picks:

            station = pick.waveform_id.station_code
            stations.append(station)
            channel = pick.waveform_id.channel_code
            print(channel)
            location = pick.waveform_id.location_code
            location = location if location else ''

            ARC.append(
                f"ARC {station:<5} {channel:<3} {network:<2} {location:<2} {ot.year:<4} "
                f"{ot.month:0>2}{ot.day:0>2} {ot.hour:0>2}"
                f"{ot.minute:0>2} {ot.second:0>2}"
            )

        count = len(set(stations))
        stations_count.append(count)
        if count > thres:
            _write_nordic(event, wavefiles=ARC, outdir=out_dir, filename=out_file)
    return stations_count


def plot_stations_count_histogram(data, threshold, title='', fig_dir='.'):
    print("Plot Cumulative of stations count.")
    plt.figure()
    hist, bin_edges, _ = plt.hist(data, 20)
    cumulative_hist = np.cumsum(hist[::-1])[::-1]
    plt.axvline(np.mean(data), color='gray', label='mean', linestyle='--')
    plt.axvline(np.median(data), color='k', label='median', linestyle='--')

    plt.plot(bin_edges[:-1], cumulative_hist, drawstyle='steps-post', color='b',
             label='Cumulative counts')
    plt.axvline(threshold, color='r', label='threshold', linestyle='--')
    plt.xlabel("#Stations")
    plt.ylabel("Cumulative Count")
    event_sum = len(data)
    good_event_sum = sum(1 for i in data if i > threshold)
    plt.title(f'{title}\n{good_event_sum}/{event_sum} (#stations>{threshold} / all)')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f'{title}.png'), dpi=300)


if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['global']
    file = "hyp.out"
    thres = 20

    stations_count = analysis_distribution_of_stations(c['events_dir'],
                                                       file,
                                                       thres,
                                                       network=c['network'],
                                                       out_dir=c['events_dir'])
    plot_stations_count_histogram(stations_count,
                                  threshold=thres,
                                  title='cumulative_stations_count',
                                  fig_dir=f'./figure/{c["dataset_name"]}')
