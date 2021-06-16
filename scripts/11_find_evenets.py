import seisnn
from obspy import UTCDateTime
from obspy.core.event import (
    Pick, Event, Origin, WaveformStreamID, EventDescription)
from obspy.io.nordic import core
from datetime import datetime


def split_event(db, from_time, to_time, station_limit=2):
    station = []
    ARC = []
    picks_query = db.get_picks(from_time=from_time, to_time=to_time,
                               tag='predict')
    event_time = UTCDateTime(from_time)
    for pick in picks_query:
        if pick.station not in station:
            station.append(pick.station)
            channel = db.get_waveform(
                station=pick.station,
                from_time=from_time,
                to_time=to_time
            )
            channel = ['EHZ','EHN','EHE']

            network = db.get_inventories(station=pick.station)
            network = network[0].network
            location = ''

            for chan in channel:
                ARC.append(
                    f'ARC {pick.station:<5} {chan:<3} {network:<2} {location:<2} {event_time.year:<4} {event_time.month:0>2}'
                    f'{event_time.day:0>2} {event_time.time.hour:0>2}{event_time.time.minute:0>2} {event_time.time.second:0>2} {int(UTCDateTime(to_time) - UTCDateTime(from_time)):<5}')
    if len(station) >= station_limit:
        return picks_query, ARC
    else:
        return [], []


def write_events(db, start, end):
    picks_query, ARC = split_event(db, start, end)
    ev = Event()
    ev.event_descriptions.append(EventDescription())
    ev.origins.append(Origin(
        time=UTCDateTime(start),
        latitude=0,
        longitude=0,
        depth=0))
    for p in picks_query:
        _waveform_id_1 = WaveformStreamID(
            station_code=p.station,
            channel_code='EHZ',
            network_code='HL'
        )
        ev.picks.append(
            Pick(waveform_id=_waveform_id_1,
                 phase_hint=p.phase,
                 time=UTCDateTime(p.time),
                 evaluation_mode="automatic"
                 )
        )
    return ev, ARC


def picks_to_sfile(database, from_time, to_time, duration_time=15):
    db = seisnn.sql.Client(database=database)
    total_time = UTCDateTime(to_time) - UTCDateTime(from_time)
    start = from_time
    end = to_time

    for i in range(int(total_time / duration_time) + 1):
        if i == int(total_time / duration_time) + 1:
            ev, ARC = write_events(db, start, end)
        else:
            end = str(datetime.strptime(
                str(UTCDateTime(start) + duration_time),
                '%Y-%m-%dT%H:%M:%S.%fZ'))
            ev, ARC = write_events(db, start, end)
            start = end
        if not ARC == []:
            core._write_nordic(ev, filename=None, wavefiles=ARC,
                               outdir='/home/andy/predict_sfile')


database = 'Hualien.db'
from_time = '2019-05-10 12:30:00'
to_time = '2019-05-12 12:30:00'
picks_to_sfile(database, from_time, to_time,duration_time=300)
