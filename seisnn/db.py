"""
SQL Database
=============

SQLite database for metadata.

.. autosummary::
    :toctree: database

    Geometry
    Picks
    Client

"""

import os
from operator import attrgetter
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, BigInteger, ForeignKey, String, DateTime, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from seisnn.utils import get_config

Base = declarative_base()


class Geometry(Base):
    """Geometry table for sql database."""
    __tablename__ = 'geometry'
    network = Column(String, nullable=False)
    station = Column(String, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)

    def __init__(self, net, sta, loc):
        self.network = net
        self.station = sta
        self.latitude = loc['latitude']
        self.longitude = loc['longitude']
        self.elevation = loc['elevation']

    def __repr__(self):
        return f"Geometry(" \
               f"Network={self.network}, " \
               f"Station={self.station}, " \
               f"Latitude={self.latitude:>7.4f}, " \
               f"Longitude={self.longitude:>8.4f}, " \
               f"Elevation={self.elevation:>6.1f})"

    def add_db(self, session):
        session.add(self)


class Event(Base):
    """Event table for sql database."""
    __tablename__ = 'event'
    id = Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    time = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth = Column(Float, nullable=False)

    def __init__(self, event):
        self.time = event.origins[0].time.datetime
        self.latitude = event.origins[0].latitude
        self.longitude = event.origins[0].longitude
        self.depth = event.origins[0].depth

    def __repr__(self):
        return f"Event(" \
               f"Time={self.time}" \
               f"Latitude={self.latitude:>7.4f}, " \
               f"Longitude={self.longitude:>8.4f}, " \
               f"Depth={self.depth:>6.1f})"

    def add_db(self, session):
        session.add(self)


class Pick(Base):
    """Pick table for sql database."""
    __tablename__ = 'pick'
    id = Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    time = Column(DateTime, nullable=False)
    station = Column(String, ForeignKey('geometry.station'), nullable=False)
    phase = Column(String, nullable=False)
    tag = Column(String, nullable=False)
    snr = Column(Float)

    def __init__(self, pick, tag):
        self.time = pick.time.datetime
        self.station = pick.waveform_id.station_code
        self.phase = pick.phase_hint
        self.tag = tag

    def __repr__(self):
        return f"Pick(" \
               f"Time={self.time}, " \
               f"Station={self.station}, " \
               f"Phase={self.phase}, " \
               f"Tag={self.tag}, " \
               f"SNR={self.snr})"

    def add_db(self, session):
        session.add(self)


class TFRecord(Base):
    """TFRecord table for sql database."""
    __tablename__ = 'tfrecord'
    id = Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    file = Column(String)
    tag = Column(String)
    station = Column(String, ForeignKey('geometry.station'))

    def __init__(self, tfrecord):
        pass

    def __repr__(self):
        return f"TFRecord(" \
               f"File={self.file}, " \
               f"Tag={self.tag}, " \
               f"Station={self.station})"

    def add_db(self, session):
        session.add(self)


class Waveform(Base):
    """Waveform table for sql database."""
    __tablename__ = 'waveform'
    id = Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    starttime = Column(DateTime, nullable=False)
    endtime = Column(DateTime, nullable=False)
    station = Column(String, ForeignKey('geometry.station'))
    tfrecord = Column(String, ForeignKey('tfrecord.file'))

    def __init__(self, waveform):
        pass

    def __repr__(self):
        return f"Waveform(" \
               f"Start Time={self.starttime}, " \
               f"End Time={self.endtime}, " \
               f"Station={self.station}, " \
               f"TFRecord={self.tfrecord})"

    def add_db(self, session):
        session.add(self)


class Client:
    """A client for manipulate sql database"""

    def __init__(self, database, echo=False):
        config = get_config()
        db_path = os.path.join(config['DATABASE_ROOT'], database)
        self.engine = create_engine(f'sqlite:///{db_path}?check_same_thread=False', echo=echo)
        Base.metadata.create_all(bind=self.engine)
        self.session = sessionmaker(bind=self.engine)

    def add_geom(self, geom, network):
        with self.session_scope() as session:
            counter = 0
            for sta, loc in geom.items():
                Geometry(network, sta, loc).add_db(session)
                counter += 1
            session.commit()
            print(f'Input {counter} stations')

    def get_geom(self, station=None, network=None):
        with self.session_scope() as session:
            query = session.query(Geometry)
            if station:
                station = self.replace_sql_wildcard(station)
                query = query.filter(Geometry.station.like(station))
            if network:
                network = self.replace_sql_wildcard(network)
                query = query.filter(Geometry.network.like(network))

        return query

    def geom_summery(self):
        with self.session_scope() as session:
            station = session.query(Geometry.station).order_by(Geometry.station)
            station_count = session.query(Geometry.station).count()
            print(f'Station name:')
            print([stat[0] for stat in station], '\n')
            print(f'Total {station_count} stations\n')

            boundary = session.query(func.min(Geometry.longitude), func.max(Geometry.longitude),
                                     func.min(Geometry.latitude), func.max(Geometry.latitude)).all()
            print(f'Station boundary:')
            print(f'West: {boundary[0][0]:>8.4f}')
            print(f'East: {boundary[0][1]:>8.4f}')
            print(f'South: {boundary[0][2]:>7.4f}')
            print(f'North: {boundary[0][3]:>7.4f}\n')

    def plot_map(self):
        from seisnn.plot import plot_map
        with self.session_scope() as session:
            geometry = session.query(Geometry.latitude, Geometry.longitude, Geometry.network).all()
            events = session.query(Event.latitude, Event.longitude).all()

        plot_map(geometry, events)

    def add_events(self, events, tag, remove_duplicates=True):
        with self.session_scope() as session:
            event_count = 0
            pick_count = 0
            for event in events:
                Event(event).add_db(session)
                event_count += 1
                for pick in event.picks:
                    Pick(pick, tag).add_db(session)
                    pick_count += 1

            print(f'Input {event_count} events, {pick_count} picks')

        if remove_duplicates:
            self.remove_duplicates(Event, ['time', 'latitude', 'longitude', 'depth'])
            self.remove_duplicates(Pick, ['time', 'phase', 'station', 'tag'])

    def get_picks(self, starttime=None, endtime=None,
                  station=None, phase=None, tag=None):
        with self.session_scope() as session:
            query = session.query(Pick)
            if starttime:
                query = query.filter(Pick.time >= starttime)
            if endtime:
                query = query.filter(Pick.time <= endtime)
            if station:
                station = self.replace_sql_wildcard(station)
                query = query.filter(Pick.station.like(station))
            if phase:
                query = query.filter(Pick.phase.like(phase))
            if tag:
                query = query.filter(Pick.tag.like(tag))

        return query

    def event_summery(self):
        with self.session_scope() as session:
            time = session.query(func.min(Event.time), func.max(Event.time)).all()
            print(f'Event time duration:')
            print(f'From: {time[0][0].isoformat()}')
            print(f'To:   {time[0][1].isoformat()}\n')

            event_count = session.query(Event).count()
            print(f'Total {event_count} events\n')

            boundary = session.query(func.min(Event.longitude), func.max(Event.longitude),
                                     func.min(Event.latitude), func.max(Event.latitude)).all()
            print(f'Event boundary:')
            print(f'West: {boundary[0][0]:>8.4f}')
            print(f'East: {boundary[0][1]:>8.4f}')
            print(f'South: {boundary[0][2]:>7.4f}')
            print(f'North: {boundary[0][3]:>7.4f}\n')
            self.pick_summery()

    def pick_summery(self):
        with self.session_scope() as session:
            time = session.query(func.min(Pick.time), func.max(Pick.time)).all()
            print(f'Pick time duration:')
            print(f'From: {time[0][0].isoformat()}')
            print(f'To:   {time[0][1].isoformat()}\n')

            print(f'Phase count:')
            phase_group_count = session.query(Pick.phase, func.count(Pick.phase)) \
                .group_by(Pick.phase).all()
            ps_picks = 0
            for phase, count in phase_group_count:
                if phase in ['P', 'S']:
                    ps_picks += count
                print(f'{count} "{phase}" picks')
            print(f'Total {ps_picks} P + S picks\n')

            station_count = session.query(Pick.station.distinct()).count()
            print(f'Picks cover {station_count} stations:')

            station = session.query(Pick.station.distinct()).order_by(Pick.station).all()
            print([stat[0] for stat in station], '\n')

            no_pick_station = session.query(Geometry.station)\
                .order_by(Geometry.station) \
                .filter(Geometry.station.notin_(session.query(Pick.station))).all()
            if no_pick_station:
                print(f'{len(no_pick_station)} stations without picks:')
                print([stat[0] for stat in no_pick_station], '\n')

            no_geom_station = session.query(Pick.station.distinct())\
                .order_by(Pick.station) \
                .filter(Pick.station.notin_(session.query(Geometry.station))).all()
            if no_geom_station:
                print(f'{len(no_geom_station)} stations without geometry:')
                print([stat[0] for stat in no_geom_station], '\n')

    def remove_duplicates(self, table, match_columns: list):
        with self.session_scope() as session:
            attrs = attrgetter(*match_columns)
            table_columns = attrs(table)
            distinct = session.query(table, func.min(table.id)) \
                .group_by(*table_columns)
            duplicate = session.query(table) \
                .filter(table.id.notin_(distinct.with_entities(table.id))) \
                .delete(synchronize_session='fetch')
            print(f'Remove {duplicate} duplicate {table.__tablename__}s')

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.session()
        try:
            yield session
            session.commit()
        except IntegrityError as err:
            print(f'Error: {err.orig}')
            session.rollback()
        finally:
            session.close()

    @staticmethod
    def replace_sql_wildcard(string):
        string = string.replace('?', '_')
        string = string.replace('*', '%')
        return string


if __name__ == "__main__":
    pass
