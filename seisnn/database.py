import os
from sqlalchemy import create_engine, Column, Integer, BigInteger, ForeignKey, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from seisnn.utils import get_config

config = get_config()
db_path = os.path.join(config['DATABASE_ROOT'], 'test.db')

Base = declarative_base()


class Geometry(Base):
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
    __tablename__ = 'picks'
    id = Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    time = Column(DateTime, nullable=False)
    station = Column(String, ForeignKey('geometry.station'))
    phase = Column(String, nullable=False)
    name = Column(String, nullable=False)
    snr = Column(Float)
    err = Column(Float)

    def __init__(self, sta, pick):
        self.time = pick.time.datetime
        self.station = sta
        self.phase = pick.phase_hint
        self.name = 'manual'

    def add_db(self, session):
        session.add(self)


engine = create_engine(f'sqlite:///{db_path}', echo=True)
Base.metadata.create_all(bind=engine)


def db_session(func):
    def wrapper(*args):
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            func(*args, session)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    return wrapper


@db_session
def add_geom(geom, session):
    for sta, loc in geom.items():
        Geometry(sta, loc).add_db(session)

@db_session
def add_picks(pick_dict, session):
    for sta, picks in pick_dict.items():
        for pick in picks:
            Picks(sta, pick).add_db(session)