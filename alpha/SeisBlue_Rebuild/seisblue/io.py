# -*- coding: utf-8 -*-
from seisblue import utils, core

import yaml
import os
import h5py
from tqdm import tqdm
from jinja2 import Template
import numpy as np
import warnings
import obspy
import glob
import pandas as pd
import io
from datetime import datetime

from obspy.core import inventory
from obspy import Stream, UTCDateTime
from obspy.core.event import (Event, EventDescription, Origin, Pick,
                              WaveformStreamID, Magnitude, Arrival, ResourceIdentifier,
                              QuantityError)
from obspy.core.event.source import (FocalMechanism, NodalPlanes, NodalPlane)
from obspy.io.nordic.core import _write_nordic, read_nordic


def get_obspy_event(filepath, debug=False):
    """
    Returns obspy.event list from sfile.
    :param str filepath: Sfile file path.
    :param bool debug: If False, warning from reader will be ignore,
        default to False.
    :rtype: list[obspy.core.event.event.Event]
    :return: List of obspy events.
    """
    with warnings.catch_warnings():
        if not debug:
            warnings.simplefilter("ignore")
        try:
            catalog = obspy.io.nordic.core.read_nordic(filepath)
            events = catalog.events
            for event in events:
                event.comments.append(obspy.core.event.Comment(text=filepath))
            return events

        except Exception as err:
            if debug:
                print(err)


def get_obspy_events(events_dir, filename="*.out"):
    """
    Returns obspy events from events directory.
    :param str events_dir: Directory contains SEISAN sfile.
    :rtype: list[obspy.core.event.event.Event]
    :return: List of obspy events.
    """
    events_path_list = glob.glob(os.path.join(events_dir, "*L.S*"))
    if len(events_path_list) == 0:
        events_path_list = glob.glob(os.path.join(events_dir, filename))
    obspy_events = utils.parallel(events_path_list, func=get_obspy_event)
    obspy_events = [event for events in obspy_events if events for event in
                    events if event.origins[0].latitude]
    obspy_events.sort(key=lambda event: event.origins[0].time)
    print(f"Read {len(obspy_events)} events from {events_dir}.")
    return obspy_events


def read_yaml(filepath):
    with open(filepath, 'r') as file:
        config_raw = file.read()
    config = yaml.safe_load(config_raw)
    vars = Template(config_raw).render(config)
    result = yaml.safe_load(vars)
    result = dict(map(lambda item: (item[0], evaluate_string(item[1])),
                      result.items()))
    return result


def evaluate_string(value):
    try:
        return eval(value)
    except Exception as e:
        return value


def write_hdf5_layer(group, dictionary):
    """
    Write HDF5 dynamically.
    :param h5py._hl.group.Group group: group
    :param dict[any, any] dictionary:
    """
    for key, value in dictionary.items():
        try:
            if value is None:
                continue
            elif isinstance(value, dict):
                sub_group = enter_hdf5_sub_group(group, key)
                write_hdf5_layer(sub_group, value)
            elif isinstance(value, list):
                sub_group = enter_hdf5_sub_group(group, key)
                for i, item in enumerate(value):
                    if sub_group.name.split('/')[-1] == 'picks':
                        min_group = enter_hdf5_sub_group(sub_group,
                                                         f'{item["phase"]}')
                    else:
                        if "id" in item.keys():
                            min_group = enter_hdf5_sub_group(sub_group, f'{item["id"]}')
                        elif "tag" in item.keys():
                            min_group = enter_hdf5_sub_group(sub_group, f'{item["tag"]}')
                        elif "channel" in item.keys():
                            min_group = enter_hdf5_sub_group(sub_group,
                                                        f'{item["channel"]}')
                        else:
                            min_group = enter_hdf5_sub_group(sub_group, f'{key}{i}')
                    write_hdf5_layer(min_group, item)
            elif isinstance(value, np.ndarray):
                if 'data' not in group.keys():
                    group.create_dataset(key, data=value)
            elif isinstance(value, (int, float, str, np.integer)):
                if isinstance(value, np.integer):
                    value = value.item()
                group.attrs.create(key, value)
            else:
                group.attrs.create(key, value.isoformat())
        except Exception as e:
            print(f"key={key},\n error={e}")


def enter_hdf5_sub_group(group, key):
    if key not in group.keys():
        sub_group = group.create_group(key)
    else:
        sub_group = group[key]
    return sub_group


def del_hdf5_layer(group):
    for key in list(group.keys()):
        del group[key]

    for attr in list(group.attrs.keys()):
        del group.attrs[attr]


def read_hypout(hypo_result):
    picks = []
    azm = []
    origin = obspy.core.event.origin.Origin()
    origin["quality"] = obspy.core.event.origin.OriginQuality()
    origin["origin_uncertainty"] = obspy.core.event.origin.OriginUncertainty()
    f = io.BytesIO(hypo_result.stdout)
    f = f.readlines()

    skip = False
    for i, l in enumerate(f):
        if l.decode("ascii")[2:6] == "date":
            skip = i
    if not skip:
        return origin, picks

    f = io.BytesIO(hypo_result.stdout)
    #  date hrmn   sec      lat      long depth   no m    rms  damp erln erlt erdp
    # 21 420 2358 51.29 2351.54N 121 32.6E   3.9    6 3   0.19 0.000  6.6 13.2574.0

    hyp_event = pd.read_fwf(
        f,
        skiprows=skip,
        nrows=1,
        colspecs=[
            (0, 6),
            (7, 9),
            (9, 11),
            (11, 17),
            (17, 26),
            (26, 36),
            (36, 42),
            (42, 47),
            (47, 49),
            (49, 56),
            (56, 62),
            (62, 67),
            (67, 72),
            (72, 77),
        ],
    )
    f = io.BytesIO(hypo_result.stdout)
    # stn   dist   azm  ain w phas    calcphs hrmn tsec  t-obs  t-cal    res   wt di
    # SF64     9 218.3 58.6 0 P    A  PN3     2358 54.0   2.75   2.47   0.28 1.00 11
    hyp_pick = pd.read_fwf(
        f,
        skiprows=skip + 2,
        skipfooter=4,
        colspecs=[
            (0, 6),
            (6, 11),
            (11, 17),
            (17, 22),
            (22, 24),
            (25, 29),
            (30, 31),
            (33, 40),
            (41, 45),
            (46, 50),
            (50, 57),
            (57, 64),
            (64, 71),
            (71, 76),
            (76, 79),
        ],
    )
    for index, row in hyp_event.iterrows():
        lat, lon = convert_lat_lon(row["lat"], row["long"])
        origin.depth = row["depth"] * 1000
        origin.depth_errors.uncertainty = row["erdp"]
        origin.latitude = lat
        origin.latitude_errors.uncertainty = row["erlt"]
        origin.longitude = lon
        origin.longitude_errors.uncertainty = row["erln"]
        origin.quality.used_station_count = row["no"]
        origin.quality.azimuthal_gap = 0
        origin.time_errors = row["rms"]
        if float(row["sec"]) == 60:
            row["mn"] = int(str(row["mn"]).strip()) + 1
            row["sec"] = 0
        if float(row["mn"]) == 60:
            row["hr"] = int(str(row["hr"]).strip()) + 1
            row["mn"] = 0

        origin.time = obspy.UTCDateTime(
            int("20" + row["date"][0:2].strip()),
            int(row["date"][2:4].strip()),
            int(row["date"][4:6].strip()),
            int(str(row["hr"]).strip()),
            int(str(row["mn"]).strip()),
            float(row["sec"]),
        )
    for index, row in hyp_pick.iterrows():
        try:
            pick = Pick(
                waveform_id=WaveformStreamID(station_code=row["stn"]),
                phase_hint=row["phas"],
                time=origin.time + float(row["t-obs"]),
                evaluation_mode="automatic",
                time_errors=row["res"],
            )

            arrival = obspy.core.event.origin.Arrival(
                pick_id=pick.resource_id, phase=row["phas"], time_residual=row["res"]
            )
        except Exception as err:

            continue
        picks.append(pick)
        origin.arrivals.append(arrival)
        azm.append(float(row["azm"]))
    azm.sort(reverse=True)
    try:
        for i in range(len(azm)):
            if i == 0:
                gap = int(abs(azm[i] - (azm[i - 1] + 360)))
            else:
                gap = int(abs(azm[i] - azm[i - 1]))
            if gap > origin.quality.azimuthal_gap:
                origin.quality.azimuthal_gap = gap
    except TypeError as err:
        print(err)
        print(azm[i], "   ", azm[i - 1])
    return origin, picks


def convert_lat_lon(lat, lon):
    NS = 1
    if lat[-1] == "S":
        NS = -1

    EW = 1
    if lon[-1] == "W":
        EW = -1
    if "." in lat[0:4]:
        lat_degree = int(lat[0:1])
        lat_minute = float(lat[1:-1]) / 60
    else:
        lat_degree = int(lat[0:2])
        lat_minute = float(lat[2:-1]) / 60
    if "." not in lat:  # high accuracy lat-lon
        lat_minute /= 1000
    lat = (lat_degree + lat_minute) * NS
    lat = inventory.util.Latitude(lat)

    lon_degree = int(lon[0:3])
    lon_minute = float(lon[3:-1]) / 60
    if "." not in lon:  # high accuracy lat-lon
        lon_minute /= 1000
    lon = (lon_degree + lon_minute) * EW
    lon = inventory.util.Longitude(lon)
    return (
        lat,
        lon,
    )


def read_hdf5(instance):
    labels = []
    dict = instance['timewindow'].attrs

    timewindow = core.TimeWindow(**dict)
    if 'starttime' in dict:
        timewindow.starttime = datetime.fromisoformat(timewindow.starttime)
    if 'endtime' in dict:
        timewindow.endtime = datetime.fromisoformat(timewindow.endtime)

    inventory = core.Inventory(**instance['inventory'].attrs)

    stream = core.Stream(**instance['features'].attrs)
    stream.data = np.array(instance['features'].get('data'))

    for label_h5 in instance['labels'].values():
        label = core.Label(**label_h5.attrs)
        label.data = np.array(label_h5.get('data'))
        labels.append(label)

    instance = core.Instance(
        inventory=inventory,
        timewindow=timewindow,
        features=stream,
        labels=labels,
    )

    return instance


def read_hdf5_file(filename):
    instances = []
    with h5py.File(filename, 'r') as f:
        for id, instance_h5 in f.items():
            instances.append(read_hdf5(instance_h5))
    return instances