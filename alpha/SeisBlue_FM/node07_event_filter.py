# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
import numpy
from pathlib import Path
import shutil
import pandas as pd
from obspy.io.nordic.core import _write_nordic
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
import matplotlib.cm as cm
import seaborn as sns

from seisblue import io, tool
from node06_Kagan_test import kagan_test


def get_gaps(events):
    df = pd.DataFrame(columns=["gap_azi", "gap_ain"])
    for i, event in enumerate(events):
        gap_azi = event.origins[0].quality.azimuthal_gap

        ain = []
        for pick in event.picks:
            if pick.phase_hint == 'P':
                arrival = \
                    [arr for arr in event.origins[0].arrivals if
                     arr.pick_id == pick.resource_id][0]
                ain.append(int(arrival.takeoff_angle))
        gap_ain = max(ain) - min(ain)

        df.loc[i] = [gap_azi, gap_ain]
    return df


def events_filter(events, thres_azi=360, thres_ain=180):
    result = []
    for event in events:
        if enough_polarity(event) and small_gap_azi(event, thres_azi) \
                and big_gap_ain(event, thres_ain):
            result.append(event)
    print(
        f'Get {len(result)} focal mechanism with AZI GAP < {thres_azi} and AIN GAP > {thres_ain}.')
    return result


def small_gap_azi(event, thres):
    gap = event.origins[0].quality.azimuthal_gap
    return True if gap < thres else False


def big_gap_ain(event, thres):
    ains = []
    for pick in event.picks:
        if pick.phase_hint == 'P':
            arrival = \
                [arr for arr in event.origins[0].arrivals if
                 arr.pick_id == pick.resource_id][0]
            ains.append(int(arrival.takeoff_angle))
    gap_ain = max(ains) - min(ains)
    return True if gap_ain > thres else False


def enough_polarity(event):
    polarities = [p.polarity for p in event.picks]
    p_num = polarities.count('positive')
    n_num = polarities.count('negative')
    return True if p_num > 3 and n_num > 3 else False


def small_error(event, thres):
    fm = event.focal_mechanisms
    if str(fm.method_id).split('/')[-1] == 'FPFIT':
        np = fm.nodal_planes.nodal_plane_1
        high_quality = (np.strike_errors.uncertainty <= thres) and (
                np.dip_errors.uncertainty <= thres) and (
                               np.rake_errors.uncertainty <= thres)
        return True if high_quality else False
    else:
        print('fm method id is not FPFIT.')

def write_sfile(events, out_dir):
    sum_fm = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for event, arc in events.values():
        _write_nordic(event, wavefiles=arc, outdir=out_dir)
        sum_fm += 1
    print(
        f'Add {sum_fm} focal mechanism to sfiles (outdir: {out_dir}).')


def copy_sfile(events, ori_dir, target_dir):
    shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir)
    for event in events:
        file = get_filepath(ori_dir, event.origins[0].time)
        shutil.copyfile(os.path.join(ori_dir, file),
                        os.path.join(target_dir, file))
    print(
        f'Add {len(events)} focal mechanism to sfiles (outdir: {target_dir}).')


def get_filepath(event_dir, evtime):
    range_list = []
    for i in range(30):  # Look +/- 30 seconds around origin time
        range_list.append(i)
        range_list.append(-1 * i)
    range_list = range_list[1:]
    for add_secs in range_list:
        sfilename = (evtime + add_secs).datetime.strftime('%d-%H%M-%S') + \
                    'L.S' + (evtime + add_secs). \
                        datetime.strftime('%Y%m')
        if (Path(event_dir) / sfilename).is_file():
            return sfilename


def plot_histogram(df, title='', fig_dir='.', out_file='hist.png',
                   thres_azi=None, thres_ain=None):
    plt.figure(figsize=(10, 6))
    plt.suptitle(title, fontsize=18)
    ax1 = plt.subplot(2, 2, 1)
    counts, edges, bars = plt.hist(df["gap_azi"], bins=20)
    plt.bar_label(bars)
    plt.twinx()
    df['gap_azi'].plot(kind='kde', color='b', label='density')
    if thres_azi:
        plt.axvline(thres_azi, color='k', linestyle='--', label='threshold')
        plt.axvspan(thres_azi, max(df['gap_azi']), color='white', alpha=0.5)
    plt.legend()
    plt.xlabel('Gap in az')
    plt.xlim(edges[0], edges[-1])
    plt.title(
        f'Gap in azimuth\n({thres_azi=})')

    plt.subplot(2, 2, 3, sharex=ax1)
    pdf = counts / sum(counts)
    cdf = numpy.cumsum(pdf)
    plt.plot(edges[1:], pdf, color="red", label="PDF")
    plt.plot(edges[1:], cdf, label="CDF")
    plt.legend()
    plt.grid()
    plt.xlabel('Gap in az')
    if thres_azi:
        plt.axvline(thres_azi, color='k', linestyle='--', label='threshold')
        plt.axvspan(thres_azi, max(df['gap_azi']), color='white', alpha=0.5)

    ax2 = plt.subplot(2, 2, 2)
    counts, edges, bars = plt.hist(df["gap_ain"], bins=20)
    plt.bar_label(bars)
    plt.twinx()
    df['gap_ain'].plot(kind='kde', color='b', label='density')
    if thres_ain:
        plt.axvline(thres_ain, color='k', linestyle='--', label='threshold')
        plt.axvspan(min(df['gap_ain']), thres_ain, color='white', alpha=0.5)
    plt.xlabel('Gap in ain')
    plt.xlim(edges[0], edges[-1])
    plt.legend()
    plt.title(f'Gap in angle of incidence \n({thres_ain=})')

    plt.subplot(2, 2, 4, sharex=ax2)
    pdf = counts / sum(counts)
    cdf = numpy.cumsum(pdf)
    plt.plot(edges[1:], pdf, color="red", label="PDF")
    plt.plot(edges[1:], cdf, label="CDF")
    plt.legend()
    plt.grid()
    plt.xlabel('Gap in ain')
    if thres_ain:
        plt.axvline(thres_ain, color='k', linestyle='--', label='threshold')
        plt.axvspan(min(df['gap_ain']), thres_ain, color='white', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, out_file), dpi=300)


def get_threshold(data, number=None, thres_probability=None):
    count, bins_count = numpy.histogram(data, bins=10)
    pdf = count / sum(count)
    cdf = numpy.cumsum(pdf)
    if number:
        thres_probability = number / len(cdf)
    elif (not thres_probability) and (not number):
        print('Give threshold of probability or number of events.')

    left = 0
    right = len(cdf) - 1
    result_index = -1

    while left <= right:
        mid = (left + right) // 2
        if cdf[mid] < thres_probability:
            result_index = mid
            left = mid + 1
        else:
            right = mid - 1
    threshold = thres_probability[result_index]
    return threshold


def plot_pivot_table(df, filepath):
    pivot_table = df.pivot_table(values='percentage', index='azi',
                                 columns='ain')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, cmap='viridis', annot=True)
    plt.xlabel('ain')
    plt.ylabel('azi')
    plt.title('percentage of kagan angle < 40')
    plt.savefig(filepath)


def analysis_percentage(obspy_events, ori_dir, target_dir):
    azi_range = range(0, 360+20, 20)
    ain_range = range(0, 180+20, 20)
    combinations = list(itertools.product(azi_range, ain_range))
    distribution_percentage = pd.DataFrame(columns=['azi', 'ain', 'percentage', 'small_count', 'all_count'])
    for i, (threshold_azi, threshold_ain) in tqdm(enumerate(combinations)):
        filtered_events = events_filter(obspy_events,
                                        thres_azi=threshold_azi,
                                        thres_ain=threshold_ain)

        tool.check_dir(target_dir, recreate=True)
        copy_sfile(filtered_events,
                   ori_dir=ori_dir,
                   target_dir=target_dir)
        filepath = f'kagan_angles_azi{threshold_azi}_ain{threshold_ain}.png'
        (small_count, all_count, small_percentage) = kagan_test(filepath, plot=False)
        distribution_percentage.loc[i] = {
            'azi': threshold_azi,
            'ain': threshold_ain,
            'percentage': small_percentage,
            'small_count': small_count,
            'all_count': all_count}

    filtered_df = distribution_percentage[distribution_percentage['percentage'] != 0]
    print(filtered_df)
    plot_pivot_table(filtered_df, os.path.join(fig_dir, 'distribution_percentage.jpg'))


if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['quality_filter']
    fig_dir = os.path.join('./figure', c['dataset_name'])
    events_dir = glob.glob(f"./result/{c['dataset_name']}/fine")[0]
    obspy_events = io.get_obspy_events(events_dir)
    df = get_gaps(obspy_events)
    tool.check_dir(fig_dir)
    ori_dir = f"./result/{c['dataset_name']}/fine"
    target_dir = f"./result/{c['dataset_name']}/fine/best"

    # analysis_percentage(obspy_events, ori_dir, target_dir)

    plot_histogram(df,
                   title=f'{c["dataset_name"]}',
                   fig_dir=fig_dir,
                   out_file='distribution_of_gap.png',
                   thres_azi=c['threshold_azi'],
                   thres_ain=c['threshold_ain'])

    filtered_events = events_filter(obspy_events,
                                    thres_azi=c['threshold_azi'],
                                    thres_ain=c['threshold_ain'])
    tool.check_dir(target_dir, recreate=True)
    copy_sfile(filtered_events,
               ori_dir=f"./result/{c['dataset_name']}/fine",
               target_dir=target_dir)
    # filepath = f'kagan_angles_azi{c["threshold_azi"]}_ain{c["threshold_ain"]}.png'
    # kagan_test(filepath, plot=True)