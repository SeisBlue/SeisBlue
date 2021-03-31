import seisnn
from obspy import UTCDateTime
from obspy.core.event import (
    Pick, Event, Origin, WaveformStreamID, EventDescription)
from obspy.io.nordic import core




def split_event(db, from_time, to_time, station_limit=2,duration = 300):
    station = []
    ARC = []

    picks_query = db.get_picks(from_time=from_time, to_time=to_time,
                               tag='predict')
    event_time = UTCDateTime(from_time)
    for pick in picks_query:
        if pick.station not in station:
            station.append(pick.station)
            channel = db.get_waveform(station=pick.station, from_time=from_time, to_time=to_time)
            channel = channel[0].channel.split(', ')

            network = db.get_inventories(station=pick.station)
            network = network[0].network
            location = ''

            for chan in channel:
                ARC.append(
                    f'ARC {pick.station:<5} {chan:<3} {network:<2} {location:<2} {event_time.year:<4} {event_time.month:0>2}'
                    f'{event_time.day:0>2} {event_time.time.hour:0>2}{event_time.time.minute:0>2} {event_time.time.second:0>2} {duration:<5}')
    if len(station) >= station_limit:
        return picks_query,ARC


def write_events(database, from_time, to_time):
    db = seisnn.sql.Client(database=database)
    picks_query, ARC = split_event(db, from_time, to_time)
    ev = Event()
    ev.event_descriptions.append(EventDescription())
    ev.origins.append(Origin(
        time=UTCDateTime(from_time), latitude=0, longitude=0,
        depth=0))
    for p in picks_query:
        _waveform_id_1 = WaveformStreamID(station_code=p.station,
                                          channel_code='EHZ',
                                          network_code='HL')
        ev.picks.append(
            Pick(waveform_id=_waveform_id_1, phase_hint='I'+p.phase,
                 time=UTCDateTime(p.time),evaluation_mode = "automatic" )
        )
    return ev,ARC

database = 'Hualien.db'
from_time = '2019-04-19 12:30:00'
to_time = '2019-04-19 12:45:00'

ev,ARC = write_events(database, from_time, to_time)
core._write_nordic(ev, 't',wavefiles=ARC)

