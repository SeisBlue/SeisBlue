# -*- coding: utf-8 -*-
from seisblue import io, utils, SQL, core
from typing import List
import argparse
from datetime import datetime
from itertools import chain


def get_event(obspy_event):
    """
    Returns event objects from obspy events .
    :param list[obspy.core.event.event.Event] obspy_events: List of obspy events.
    :rtype: list[core.Event]
    :return: List of event objects.
    """
    origin_time = obspy_event.origins[0].time
    latitude = obspy_event.origins[0].latitude
    longitude = obspy_event.origins[0].longitude
    depth = obspy_event.origins[0].depth
    magnitude = obspy_event.magnitudes[0].mag if len(obspy_event.magnitudes) > 0 else None

    try:
        np1 = obspy_event.focal_mechanisms[0].nodal_planes.nodal_plane_1
        np = core.NodalPlane(strike=np1.strike,
                             strike_errors=np1.strike_errors.uncertainty,
                             dip=np1.dip,
                             dip_errors=np1.dip_errors.uncertainty,
                             rake=np1.rake,
                             rake_errors=np1.rake_errors.uncertainty)
    except Exception as e:
        np = None

    event = core.Event(time=datetime.utcfromtimestamp(origin_time.timestamp),
                       latitude=latitude,
                       longitude=longitude,
                       depth=depth,
                       magnitude=magnitude,
                       nodal_plane=np)
    event_SQL = core.EventSQL(event)
    return event_SQL


def get_picks(obspy_event: list, tag: str, network=None) -> List[object]:
    """
    Returns list of picks dataclass from events list.
    :param list obspy_events: Obspy Event.
    :param str tag: Pick tag.
    :rtype: list
    :return: Dataclass Pick.
    """
    picks = []
    for pick in obspy_event.picks:
        time = pick.time
        station = pick.waveform_id.station_code
        network = pick.waveform_id.network_code or network
        phase = pick.phase_hint
        polarity = None if phase == 'S' else (pick.polarity or 'undecidable')
        pick = core.Pick(time=datetime.utcfromtimestamp(time.timestamp),
                         inventory=core.Inventory(station=station, network=network),
                         phase=phase,
                         tag=tag,
                         polarity=polarity,
                         )
        pick_SQL = core.PickSQL(pick)
        picks.append(pick_SQL)
    return picks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = io.read_yaml(args.data_config_filepath)
    c = config['process_event']

    print("Processing events and picks.")
    obspy_events = io.get_obspy_events(c['events_dir'])

    events = utils.parallel(obspy_events,
                            func=get_event)
    picks = utils.parallel(obspy_events,
                           func=get_picks,
                           tag=c['tag'],
                           network=c['network'])
    picks = list(chain.from_iterable(sublist for sublist in picks if sublist))
    client = SQL.Client(c['database'], build=c['build_database'])
    client.add_events(events)
    client.add_picks(picks)

    inspector = SQL.DatabaseInspector(c['database'])
    inspector.event_summery()
    inspector.pick_summery()
