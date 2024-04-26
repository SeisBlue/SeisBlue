"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

# from src.seisblue_pipeline.seisblue import core
import obspy.io.nordic.core
from obspy import UTCDateTime
import obspy.core.event.event
import datetime
import warnings
import glob
import os
import itertools
import logging
from ...seisblue import core
import collections
import pandas as pd


def _is_waveform_dir(directory):
    return '@' not in directory and not os.path.isfile(directory)


def _is_waveform_file(file):
    return file[-1].isdigit()


def _get_station_starttime_and_endtime(files):
    station = '.'.join(files[0].split('/')[-1].split('.')[:2])
    startfile = files[0].split('/')[-1]
    endfile = files[-1].split('/')[-1]
    starttime = UTCDateTime(
        datetime.datetime.strptime(startfile[-8:], '%Y.%j'))
    endtime = UTCDateTime(
        datetime.datetime.strptime(endfile[-8:], '%Y.%j'))
    time_window = core.TimeWindow(station=station,
                                  starttime=starttime,
                                  endtime=endtime)
    return time_window


def get_stations_time_window(waveforms_dirpath):
    """
    Returns list of time window from the filename of waveforms.

    :param str waveforms_dirpath: Path of waveforms directory.
    :rtype: dict[str, core.TimeWindow]
    :return: Dictionary contains station and TimeWindow object.
    """
    log = logging.getLogger(__name__)
    directories = glob.glob(os.path.join(waveforms_dirpath, '????/??/*/*'))
    directories = list(filter(_is_waveform_dir, directories))
    stations_time_window = {}
    for directory in directories:
        files = sorted(glob.glob(os.path.join(directory, '*')))
        files = list(filter(_is_waveform_file, files))
        time_window = _get_station_starttime_and_endtime(files)
        stations_time_window[time_window.station] = time_window

    log.debug(
        f'Get {len(stations_time_window)} stations with time window from {waveforms_dirpath}')
    return stations_time_window


def _degree_minute_to_signed_decimal_degree(degree_minute):
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


def read_hyp(hyp, stations_time_window):
    """
    Returns geometry from STATION0.HYP file.

    :param str hyp: STATION0.HYP content.
    :param dict[str, core.TimeWindow] stations_time_window:
        Dictionary contains station and TimeWindow object.
    :rtype: list[core.Inventory]
    :return: List of geometry inventory object.
    """
    log = logging.getLogger(__name__)
    inventories = []
    blank_line = 0
    count = 0
    lines = hyp.split("\n")
    network = lines[-1]

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
            sta = str(line[1:6].strip())
            name = network + '.' + sta
            if name in stations_time_window.keys():
                count += 1
                time_window = stations_time_window[name]
            else:
                time_window = core.TimeWindow(
                    starttime=UTCDateTime(1970, 1, 1, 0, 0),
                    endtime=UTCDateTime(9999, 1, 1, 0, 0))
            item = core.Inventory(station=sta,
                                  latitude=lat,
                                  longitude=lon,
                                  elevation=elev,
                                  network=network,
                                  time_window=time_window)
            inventories.append(item)
    log.debug(
        f'Get {len(inventories)} stations, {count} stations contain time_window')
    return inventories


def _get_obspy_event(filepath, debug=False):
    """
    Returns obspy.event list from sfile.

    :param str filepath: Sfile file path.
    :param bool debug: If False, warning from reader will be ignore,
        default to False.
    :rtype: list[obspy.core.event.event.Event]
    :return: List of obspy events.
    """
    log = logging.getLogger(__name__)
    with warnings.catch_warnings():
        if not debug:
            warnings.simplefilter("ignore")
        try:
            catalog = obspy.io.nordic.core.read_nordic(filepath)
            return catalog.events

        except Exception as err:
            if debug:
                log.error(err)


def read_sfile(events_dir):
    """
    Returns obspy events from events directory.

    :param str events_dir: Directory contains SEISAN sfile.
    :rtype: list[obspy.core.event.event.Event]
    :return: List of obspy events.
    """
    log = logging.getLogger(__name__)
    events_path_list = glob.glob(os.path.join(events_dir, "*"))
    obspy_events = list(map(_get_obspy_event, events_path_list[:200]))
    flatten = itertools.chain.from_iterable
    while isinstance(obspy_events[0], list):
        obspy_events = flatten(obspy_events)
        obspy_events = [event for event in obspy_events if
                        event.origins[0].latitude]
    log.debug(f"Get {len(obspy_events)} events from {events_dir}")
    return obspy_events


def _transfer_obspy_event_into_dataclass(obspy_event):
    origin_time = obspy_event.origins[0].time
    latitude = obspy_event.origins[0].latitude
    longitude = obspy_event.origins[0].longitude
    depth = obspy_event.origins[0].depth
    event_dataclass = core.Event(origin_time=origin_time,
                                 latitude=latitude,
                                 longitude=longitude,
                                 depth=depth)
    return event_dataclass


def get_events_from_obspy_events(obspy_events):
    """
    Returns event objects from obspy events .

    :param list[obspy.core.event.event.Event] obspy_events: List of obspy events.
    :rtype: list[core.Event]
    :return: List of event objects.
    """
    events = list(map(_transfer_obspy_event_into_dataclass, obspy_events))
    return events


def _transfer_obspy_pick_into_dataclass(pick, tag=None):
    time = pick.time
    station = pick.waveform_id.station_code
    phase = pick.phase_hint
    inventory = core.Inventory(station=station)
    pick_dataclass = core.Pick(time=time,
                               inventory=inventory,
                               phase=phase,
                               tag=tag)
    return pick_dataclass


def get_picks_from_obspy_events(obspy_events, tag):
    """
    Returns picks dataclass from obspy events .

    :param list[obspy.core.event.event.Event] obspy_events: List of obspy events.
    :param str tag: Pick tag.
    :rtype: list[core.Pick]
    :return: List of dataclass picks.
    """
    log = logging.getLogger(__name__)
    picks = []
    for event in obspy_events:
        for pick in event.picks:
            picks.append(_transfer_obspy_pick_into_dataclass(pick, tag))

    log.debug(f'Get {len(picks)} picks')
    return picks


def create_database(database, inventories, events, picks):
    """
    Create database and add inventory, events and picks into database.

    :param str database:
    :param list[core.Inventory] inventories: List of dataclass inventories.
    :param list[core.Event] events: List of dataclass events.
    :param list[core.Pick] picks: List of dataclass picks.
    """
    db = core.Client(database, echo=False, build=True)
    db.add_inventory(inventories, remove_duplicates=True)
    db.add_events(events, remove_duplicates=True)
    db.add_picks(picks, remove_duplicates=True)


def check_inventory(database):
    """
    Prints summery from inventory table.

    :param str database:
    """
    log = logging.getLogger(__name__)
    db = core.Client(database=database)

    stations = db.get_distinct_items("inventory", "station")
    latitudes = db.get_distinct_items("inventory", "latitude")
    longitudes = db.get_distinct_items("inventory", "longitude")

    log.debug(f"Station name (total {len(stations)} stations):")
    log.debug([station for station in stations])
    log.debug("Station boundary:")
    log.debug(f"West: {min(longitudes):>8.4f}")
    log.debug(f"East: {max(longitudes):>8.4f}")
    log.debug(f"South: {min(latitudes):>7.4f}")
    log.debug(f"North: {max(latitudes):>7.4f}")


def check_pick(database):
    """
    Prints summery from pick table.

    :param str database:
    """
    log = logging.getLogger(__name__)
    db = core.Client(database=database)

    times = db.get_distinct_items('pick', 'time')
    log.debug('Event time duration:')
    log.debug(f'From: {min(times).isoformat()}')
    log.debug(f'To:   {max(times).isoformat()}')

    log.debug('Phase count:')
    phases = db.get_distinct_items('pick', 'phase')
    for phase in phases:
        picks = db.get_picks(phase=phase)
        log.debug(f'{len(picks)} "{phase}" picks')

    pick_stations = db.get_distinct_items('pick', 'station')
    log.debug(f'Picks cover {len(pick_stations)} stations:')
    log.debug([station for station in pick_stations])

    no_pick_station = db.get_exclude_items('inventory', 'station',
                                           pick_stations)
    if no_pick_station:
        log.debug(f'{len(no_pick_station)} stations without picks:')
        log.debug([station for station in no_pick_station])

    inventory_station = db.get_distinct_items('inventory', 'station')
    no_inventory_station = db.get_exclude_items('pick', 'station',
                                                inventory_station)

    if no_inventory_station:
        log.debug(f'{len(no_inventory_station)} stations without geometry:')
        log.debug([station for station in no_inventory_station])
