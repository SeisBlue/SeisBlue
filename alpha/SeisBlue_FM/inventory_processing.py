# -*- coding: utf-8 -*-
from typing import List, Dict
import glob
import os
import datetime
import logging

import seisblue.io
from seisblue import core


def _is_waveform_dir(directory):
    return '@' not in directory and not os.path.isfile(directory)


def _is_waveform_file(file):
    return file[-1].isdigit()


def _get_station_starttime_and_endtime(files):
    startfile = files[0].split('/')[-1]
    endfile = files[-1].split('/')[-1]
    starttime = datetime.datetime.strptime(startfile[-8:], '%Y.%j')
    endtime = datetime.datetime.strptime(endfile[-8:], '%Y.%j')
    timewindow = core.TimeWindow(starttime=starttime,
                                 endtime=endtime)
    return timewindow


def get_stations_time_window(waveforms_dirpath):
    """
    Returns list of time window from the filename of waveforms.
    :param str waveforms_dirpath: Path of waveforms directory.
    :rtype: dict[str, core.TimeWindow]
    :return: Dictionary contains station and TimeWindow object.
    """
    directories = glob.glob(os.path.join(waveforms_dirpath, '??/*/??Z.D'))
    directories = list(filter(_is_waveform_dir, directories))
    stations_time_window = {}
    for directory in directories:
        files = sorted(glob.glob(os.path.join(directory, '*')))
        files = list(filter(_is_waveform_file, files))
        station = files[0].split('/')[-1].split('.')[1]
        time_window = _get_station_starttime_and_endtime(files)
        stations_time_window[station] = time_window
    return stations_time_window


def _degree_minute_to_signed_decimal_degree(degree_minute: str) -> float:
    sign = {
        'N': 1,
        'S': -1,
        'E': 1,
        'W': -1,
    }

    direction = degree_minute[-1]

    degree = int(degree_minute[:-6])
    minute = float(degree_minute[-6:-1]) / 60
    if '.' not in degree_minute:  # high accuracy lat-lon
        minute /= 1000
    signed_decimal_degree = (degree + minute) * sign[direction]

    return signed_decimal_degree


def _if_station_has_timewindow(name, station_time_dict):
    if name in station_time_dict.keys():
        time_window = station_time_dict[name]

        return time_window
    else:
        return None


def read_hyp(hyp_filepath: str, station_time_dict: Dict, network:str=None) -> List[object]:
    """
    Returns geometry from STATION0.HYP file.

    :param str hyp_filepath: STATION0.HYP content.
    :rtype: list
    :return: Geometry Inventory.
    """
    inventories = []
    blank_line = 0

    with open(hyp_filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        if not len(line):
            blank_line += 1
            continue

        if blank_line > 1:
            break

        elif blank_line == 1:
            lat = _degree_minute_to_signed_decimal_degree(line[6:14])
            lon = _degree_minute_to_signed_decimal_degree(line[14:23])
            elev = float(line[23:])
            sta = line[1:6].strip()
            timewindow = station_time_dict[sta] if sta in station_time_dict.keys() else None
            inventory = core.Inventory(station=sta,
                                       network=network,
                                       latitude=lat,
                                       longitude=lon,
                                       elevation=elev,
                                       timewindow=timewindow)
            inventories.append(inventory)
    return inventories


if __name__ == '__main__':
    config = seisblue.io.read_yaml('./config/data_config.yaml')
    c = config['process_inventory']

    station_time_dict = get_stations_time_window(c['sub_waveforms_dir'])
    inventory = read_hyp(c['hyp_filepath'], station_time_dict, c['network'])





