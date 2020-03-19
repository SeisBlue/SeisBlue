"""
Database
=============

"""

import os
from sqlalchemy import create_engine, Column, Integer, BigInteger, ForeignKey, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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

    def __init__(self, sta, pick, pickset):
        self.time = pick.time.datetime
        self.station = sta
        self.phase = pick.phase_hint
        self.pickset = pickset

    def add_db(self, session):
        session.add(self)


class Client:
    """A client for manipulate sql database"""
    def __init__(self, database, echo=False):
        config = get_config()
        db_path = os.path.join(config['DATABASE_ROOT'], f'{database}.db')
        self.engine = create_engine(f'sqlite:///{db_path}', echo=echo)
        Base.metadata.create_all(bind=self.engine)
        self.session = sessionmaker(bind=self.engine)

    def add_geom(self, geom):
        session = self.session()
        try:
            for sta, loc in geom.items():
                Geometry(sta, loc).add_db(session)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def add_picks(self, pick_dict, pickset):
        session = self.session()
        try:
            for sta, picks in pick_dict.items():
                for pick in picks:

                    Picks(sta, pick, pickset).add_db(session)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

if __name__ == "__main__":
    pass
