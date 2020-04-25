"""
Database
=============

SQLite database for metadata.

.. autosummary::
    :toctree: database

    Geometry
    Picks
    Client

"""

import os
from sqlalchemy import create_engine, Column, Integer, BigInteger, ForeignKey, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from seisnn.utils import get_config

Base = declarative_base()


class Geometry(Base):
    """Geometry table for sql database."""
    __tablename__ = 'geometry'
    station = Column(String, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)

    def __init__(self, sta, loc):
        self.station = sta
        self.latitude = loc['latitude']
        self.longitude = loc['longitude']
        self.elevation = loc['elevation']

    def __repr__(self):
        return f"Geometry(Station={self.station}, " \
               f"Latitude={self.latitude:>7.4f}, " \
               f"Longitude={self.longitude:>8.4f}, " \
               f"Elevation={self.elevation:>6.1f})"

    def add_db(self, session):
        session.add(self)


class Picks(Base):
    """Picks table for sql database."""
    __tablename__ = 'picks'
    id = Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    time = Column(DateTime, nullable=False)
    station = Column(String, ForeignKey('geometry.station'))
    phase = Column(String, nullable=False)
    name = Column(String, nullable=False)
    snr = Column(Float)
    err = Column(Float)

    def __init__(self, pick, name):
        self.time = pick.time.datetime
        self.station = pick.waveform_id.station_code
        self.phase = pick.phase_hint
        self.name = name

    def __repr__(self):
        return f"Pick(Time={self.time}, " \
               f"Station={self.station}, " \
               f"Phase={self.phase}, " \
               f"Name={self.name}, " \
               f"SNR={self.snr}, " \
               f"ERR={self.err:})"

    def add_db(self, session):
        session.add(self)


class Client:
    """A client for manipulate sql database"""

    def __init__(self, database, echo=False):
        config = get_config()
        db_path = os.path.join(config['DATABASE_ROOT'], database)
        self.engine = create_engine(f'sqlite:///{db_path}', echo=echo)
        Base.metadata.create_all(bind=self.engine)
        self.session = sessionmaker(bind=self.engine)

    def add_geom(self, geom):
        session = self.session()
        try:
            for sta, loc in geom.items():
                Geometry(sta, loc).add_db(session)
            session.commit()
        except IntegrityError as err:
            print(f'Error: {err.orig}')
            session.rollback()
        finally:
            session.close()

    def get_geom(self, station=None):
        session = self.session()
        query = session.query(Geometry)
        if station:
            if '*' in station or '?' in station:
                station = station.replace('?', '_')
                station = station.replace('*', '%')
                query = query.filter(Geometry.station.like(station))
            else:
                query = query.filter(Geometry.station == station)
        session.close()
        return query

    def add_picks(self, events, name):
        session = self.session()
        try:
            for event in events:
                for pick in event.picks:
                    Picks(pick, name).add_db(session)
            session.commit()
        except IntegrityError as err:
            print(f'Error: {err.orig}')
            session.rollback()
        finally:
            session.close()

    def get_picks(self, starttime=None, endtime=None,
                  station=None, phase=None, name=None):
        session = self.session()
        query = session.query(Picks)
        if starttime:
            query = query.filter(Picks.time >= starttime)
        if endtime:
            query = query.filter(Picks.time <= endtime)
        if station:
            if '*' in station or '?' in station:
                station = station.replace('?', '_')
                station = station.replace('*', '%')
                query = query.filter(Picks.station.like(station))
            else:
                query = query.filter(Picks.station == station)
        if phase:
            query = query.filter(Picks.phase.like(phase))
        if name:
            query = query.filter(Picks.phase.like(name))
        session.close()
        return query

    def list_pick_phase(self):
        session = self.session()
        query = session.query(Picks.phase).distinct()
        return query


if __name__ == "__main__":
    pass
