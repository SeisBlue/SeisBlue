# -*- coding: utf-8 -*-
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
import ast

from obspy import Stream, UTCDateTime
from obspy.core.event import (Event, EventDescription, Origin, Pick,
                              WaveformStreamID, Magnitude, Arrival,
                              ResourceIdentifier,
                              QuantityError)
from obspy.core.event.source import (FocalMechanism, NodalPlanes, NodalPlane)
from obspy.io.nordic.core import _write_nordic, read_nordic

from seisblue import core
from seisblue.tool import to_dict, evaluate_string


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
                    if "id" in item.keys():
                        min_group = enter_hdf5_sub_group(sub_group,
                                                         f'{item["id"]}')
                    elif "tag" in item.keys():
                        min_group = enter_hdf5_sub_group(sub_group,
                                                         f'{item["tag"]}')
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


def del_hdf5_layer(group):
    for key in list(group.keys()):
        del group[key]

    for attr in list(group.attrs.keys()):
        del group.attrs[attr]


def write_hdf5(instances, mode, dataset_dir, **kwargs):
    """
    Add instances into HDF5 file.
    :param core.EventInstance instances: List of instances.
    :param list filepath: hdf5
    """
    for lable, instance_batch in instances:
        instance_batch = list(instance_batch)
        dataset = instance_batch[0].dataset
        filepath = os.path.join(dataset_dir, f'{dataset}_{lable}_{mode}.hdf5')
        with h5py.File(filepath, 'w') as f:
            for instance in instance_batch:
                HDFdict = to_dict(instance)
                if HDFdict['id'] not in f.keys():
                    instance_group = f.create_group(HDFdict['id'])
                    write_hdf5_layer(instance_group, HDFdict)
        print(f'Write {len(instance_batch)} data into {filepath}.')


def read_hdf5_event(file, key=None):
    event_instances = []
    with h5py.File(file, "r") as f:
        if key:
            archive = [f[key]]
        else:
            archive = f.values()
        for evt_instance_h5 in archive:
            instances = []
            for instance_h5 in evt_instance_h5['instances'].values():
                attr_dict = dict(instance_h5['Spick'].attrs)
                Spick = core.Pick(**attr_dict)
                Spick.time = UTCDateTime(Spick.time)
                labels = []
                for label_h5 in instance_h5['labels'].values():
                    attr_dict = dict(label_h5['pick'].attrs)
                    Ppick = core.Pick(**attr_dict)
                    Ppick.time = UTCDateTime(Ppick.time)
                    label = core.Label(pick=Ppick, tag=Ppick.tag)
                    labels.append(label)
                if 'inventory' not in attr_dict.keys():
                    attr_dict = dict(instance_h5['inventory'].attrs)
                    inventory = core.Inventory(**attr_dict)
                Ppick.inventory = inventory
                instance = core.Instance(Spick=Spick, labels=labels)
                instances.append(instance)

            if 'focal_mechanism' in evt_instance_h5['event'].keys():
                attr_dict = dict(
                    evt_instance_h5['event/focal_mechanism'].attrs)
                focal_mechanism = core.NodalPlane(**attr_dict)
            else:
                focal_mechanism = None
            attr_dict = dict(evt_instance_h5['event'].attrs)
            event = core.Event(focal_mechanism=focal_mechanism, **attr_dict)
            event.time = UTCDateTime(event.time)

            event_instance = core.EventInstance(event=event,
                                                instances=instances)
            event_instances.append(event_instance)
    return event_instances


def read_hdf5_instance(evt_instance_h5, index=None):
    instances = []
    if index is None:
        instance_h5_grp = list(evt_instance_h5['instances'].values())
    else:
        instance_h5_grp = [list(evt_instance_h5['instances'].values())[index]]
    for instance_h5 in instance_h5_grp:
        traces = []
        labels = []
        instance_dict = {}
        id = instance_h5.attrs['id']
        instance_dict['id'] = id
        if 'timewindow' in instance_h5.keys():
            attr_dict = dict(instance_h5['timewindow'].attrs)
            instance_dict['timewindow'] = core.TimeWindow(**attr_dict)

        if 'inventory' in instance_h5.keys():
            attr_dict = dict(instance_h5['inventory'].attrs)
            instance_dict['inventory'] = core.Inventory(**attr_dict)

        if 'traces' in instance_h5.keys():
            for trace_h5 in instance_h5['traces'].values():
                attr_dict = dict(trace_h5.attrs)
                trace = core.Trace(**attr_dict)
                trace.data = np.array(trace_h5['data'])
                traces.append(trace)
            instance_dict['traces'] = traces

        if 'labels' in instance_h5.keys():
            for label_h5 in instance_h5['labels'].values():
                attr_dict = dict(label_h5.attrs)
                label = core.Label(**attr_dict)
                attr_dict = dict(label_h5['pick'].attrs)
                label.pick = core.Pick(**attr_dict)
                label.data = np.array(label_h5.get('data'))
                labels.append(label)
            instance_dict['labels'] = labels
        if 'Spick' in instance_h5.keys():
            attr_dict = dict(instance_h5['Spick'].attrs)
            spick = core.Pick(**attr_dict)
            spick.time = UTCDateTime(spick.time)
            instance_dict['Spick'] = spick

        instance = core.Instance(**instance_dict)
        instances.append(instance)
    if 'focal_mechanism' in evt_instance_h5['event'].keys():
        attr_dict = dict(evt_instance_h5['event/focal_mechanism'].attrs)
        focal_mechanism = core.NodalPlane(**attr_dict)
    else:
        focal_mechanism = None
    attr_dict = dict(evt_instance_h5['event'].attrs)
    event = core.Event(focal_mechanism=focal_mechanism, **attr_dict)
    event.time = UTCDateTime(event.time)
    attr_dict = dict(evt_instance_h5.attrs)
    evt_instance = core.EventInstance(
        instances=instances,
        event=event,
        **attr_dict)

    return evt_instance


def enter_hdf5_sub_group(group, key):
    if key not in group.keys():
        sub_group = group.create_group(key)
    else:
        sub_group = group[key]
    return sub_group


def read_yaml(filepath):
    with open(filepath, 'r') as file:
        config_raw = file.read()
    config = yaml.safe_load(config_raw)
    vars = Template(config_raw).render(config)
    result = yaml.safe_load(vars)
    result = dict(map(lambda item: (item[0], evaluate_string(item[1])),
                      result.items()))
    return result


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
                pick_id=pick.resource_id, phase=row["phas"],
                time_residual=row["res"]
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
            return catalog.events

        except Exception as err:
            if debug:
                print(err)


def get_obspy_events(events_dir):
    """
    Returns obspy events from events directory.
    :param str events_dir: Directory contains SEISAN sfile.
    :rtype: list[obspy.core.event.event.Event]
    :return: List of obspy events.
    """
    events_path_list = glob.glob(os.path.join(events_dir, "*L.S*"))
    if len(events_path_list) == 0:
        events_path_list = glob.glob(os.path.join(events_dir, "*.out"))
    obspy_events = list(map(get_obspy_event, events_path_list))
    obspy_events = [event for events in obspy_events if events for event in
                    events]
    obspy_events.sort(key=lambda event: event.origins[0].time)
    return obspy_events


def write_sfile(evt_instance, out_dir, method_id, filename=None,
                flip_polarity=False):
    ev = Event()
    ev.event_descriptions.append(EventDescription())
    ot = UTCDateTime(evt_instance.event.time)

    ev.origins.append(
        Origin(
            time=ot,
            latitude=evt_instance.event.latitude,
            longitude=evt_instance.event.longitude,
            depth=evt_instance.event.depth,
        )
    )
    if evt_instance.event.focal_mechanism:
        ev.focal_mechanisms.append(
            FocalMechanism(
                method_id='ori',
                nodal_planes=NodalPlanes(
                    nodal_plane_1=NodalPlane(
                        strike=evt_instance.event.focal_mechanism.strike,
                        dip=evt_instance.event.focal_mechanism.dip,
                        rake=evt_instance.event.focal_mechanism.rake,
                        strike_errors=evt_instance.event.focal_mechanism.strike_errors,
                        dip_errors=evt_instance.event.focal_mechanism.dip_errors,
                        rake_errors=evt_instance.event.focal_mechanism.rake_errors)
                )
            )
        )
    ARC = []
    duration_time = 120

    for instance in evt_instance.instances:
        pred_label = [label for label in instance.labels if
                      label.tag == method_id]
        if pred_label:
            p_pick = pred_label[0].pick
            arrival = Arrival(azimuth=p_pick.azimuth,
                              takeoff_angle=p_pick.takeoff_angle,
                              pick_id=ResourceIdentifier(id=p_pick.pick_id))
            ev.origins[0].arrivals.append(arrival)

            station = instance.inventory.station
            network = instance.inventory.network
            location = ""
            channel = [tr.channel for tr in instance.traces]
            channel_Z = [ch for ch in channel if ch[-1] == 'Z'][0]

            for chan in channel:
                ARC.append(
                    f"ARC {station:<5} {chan:<3} {network:<2} {location:<2} {ot.year:<4} "
                    f"{ot.month:0>2}{ot.day:0>2} {ot.hour:0>2}"
                    f"{ot.minute:0>2} {ot.second:0>2} {duration_time}"
                )

            _waveform_id_1 = WaveformStreamID(
                station_code=station,
                channel_code=channel_Z,
                network_code=network,
            )
            # obspy issue #2848 pick.second = 0. bug
            time = UTCDateTime(p_pick.time)
            if time.second == 0 and time.microsecond == 0:
                time = time + 0.01
            flip_polarity_map = {'positive': 'negative', 'negative': 'positive',
                                 'undecidable': 'undecidable'}

            polarity = flip_polarity_map[
                p_pick.polarity] if flip_polarity else p_pick.polarity
            p_pick = Pick(
                resource_id=ResourceIdentifier(id=p_pick.pick_id),
                waveform_id=_waveform_id_1,
                phase_hint=p_pick.phase,
                time=time,
                evaluation_mode="manual",
                polarity=polarity,
            )
            ev.picks.append(p_pick)
            s_pick = instance.Spick
            if s_pick:
                s_pick = Pick(
                    resource_id=ResourceIdentifier(id=s_pick.pick_id),
                    waveform_id=_waveform_id_1,
                    phase_hint=s_pick.phase,
                    time=UTCDateTime(s_pick.time),
                    evaluation_mode="manual",
                )
                ev.picks.append(s_pick)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not ARC == []:
        _write_nordic(ev, filename=filename, wavefiles=ARC, outdir=out_dir)


def hdf5_to_sfile(filepath, out_dir, method_id, flip_polarity):
    with h5py.File(filepath, "r") as f:
        for event_instance_h5 in tqdm(f.values()):
            event_instance = read_hdf5_instance(event_instance_h5)
            write_sfile(event_instance, out_dir, method_id=method_id,
                        flip_polarity=flip_polarity)
        print(f"Write {len(f.values())} events in s-file (from {filepath})")


def hdf5_to_csv(filepath):
    with h5py.File(filepath, "r") as f:
        for event_instance_h5 in tqdm(f.values()):
            event_instance = read_hdf5_instance(event_instance_h5)
            time = event_instance.event.time
            latitude = event_instance.event.latitude
            longitude = event_instance.event.longitude
            depth = event_instance.event.depth

            for instance in event_instance.instances:
                for label in instance.labels:
                    if label.tag == 'ori':
                        pass
