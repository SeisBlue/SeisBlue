import os
import time
import pandas as pd
import logging

import sqlalchemy
import sqlalchemy.orm
from obspy.clients.filesystem import sds
from sqlalchemy import Index
# from sqlalchemy_utils import has_index

from ...seisblue import utils



Base = sqlalchemy.ext.declarative.declarative_base()


class PicksAssoc(Base):
    __tablename__ = "picks"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    sta = sqlalchemy.Column(sqlalchemy.String(5))
    net = sqlalchemy.Column(sqlalchemy.String(2))
    loc = sqlalchemy.Column(sqlalchemy.String(2))
    time = sqlalchemy.Column(sqlalchemy.DateTime)
    snr = sqlalchemy.Column(sqlalchemy.Float)
    phase = sqlalchemy.Column(sqlalchemy.String(5))
    locate_flag = sqlalchemy.Column(sqlalchemy.Boolean)
    assoc_id = sqlalchemy.Column(sqlalchemy.Integer)
    trace_id = sqlalchemy.Column(sqlalchemy.String(20))

    def __init__(self, sta, net, loc, pick_time, snr, phase=None, trace_id=None):
        self.sta = sta
        self.net = net
        self.loc = loc
        self.time = pick_time
        self.snr = snr
        self.phase = phase
        self.locate_flag = None
        self.assoc_id = None
        self.trace_id = trace_id

    def __repr__(self):
        return (
            f"Pick("
            f"{self.sta}, "
            f"{self.net}, "
            f"{self.loc}, "
            f'{self.time.isoformat("T")},'
            f"{self.phase}, "
            f"assoc_id: {self.assoc_id}, "
            f"trace_id:{self.trace_id})"
        )


class Candidate(Base):
    __tablename__ = "candidate"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    origin_time = sqlalchemy.Column(sqlalchemy.DateTime)
    sta = sqlalchemy.Column(sqlalchemy.String(5))
    weight = sqlalchemy.Column(sqlalchemy.Float)
    p_time = sqlalchemy.Column(sqlalchemy.DateTime)
    p_id = sqlalchemy.Column(sqlalchemy.Integer)
    s_time = sqlalchemy.Column(sqlalchemy.DateTime)
    s_id = sqlalchemy.Column(sqlalchemy.Integer)
    locate_flag = sqlalchemy.Column(sqlalchemy.Boolean)
    assoc_id = sqlalchemy.Column(sqlalchemy.Integer)

    def __init__(self, origin_time, sta, p_time, p_id, s_time, s_id):
        self.origin_time = origin_time
        self.sta = sta
        self.weight = None
        self.p_time = p_time
        self.s_time = s_time
        self.p_id = p_id
        self.s_id = s_id
        self.locate_flag = None
        self.assoc_id = None

    def __repr__(self):
        return (
            f"Candidate Event("
            f'{self.origin_time.isoformat("T")}, '
            f"{self.sta}, "
            f"p_id: {self.p_id}, "
            f"s_id: {self.s_id})"
        )

    def set_assoc_id(self, assoc_id, session, locate_flag):
        self.assoc_id = assoc_id
        self.locate_flag = locate_flag
        """
        Assign phases to modified picks
        """
        pick_p = session.query(PicksAssoc).filter(PicksAssoc.id == self.p_id)
        for pick in pick_p:
            pick.phase = "P"
            pick.assoc_id = assoc_id
            pick.locate_flag = locate_flag

        pick_s = session.query(PicksAssoc).filter(PicksAssoc.id == self.s_id)
        for pick in pick_s:
            pick.phase = "S"
            pick.assoc_id = assoc_id
            pick.locate_flag = locate_flag


class AssociatedEvent(Base):
    __tablename__ = "associated"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    origin_time = sqlalchemy.Column(sqlalchemy.DateTime)
    time_std = sqlalchemy.Column(sqlalchemy.Float)
    latitude = sqlalchemy.Column(sqlalchemy.Float)
    longitude = sqlalchemy.Column(sqlalchemy.Float)
    depth = sqlalchemy.Column(sqlalchemy.Float)
    nsta = sqlalchemy.Column(sqlalchemy.Integer)
    rms = sqlalchemy.Column(sqlalchemy.Float)
    erln = sqlalchemy.Column(sqlalchemy.Float)
    erlt = sqlalchemy.Column(sqlalchemy.Float)
    erdp = sqlalchemy.Column(sqlalchemy.Float)
    gap = sqlalchemy.Column(sqlalchemy.Float)

    def __init__(
            self,
            origin_time,
            time_std,
            latitude,
            longitude,
            depth,
            nsta,
            rms,
            erln,
            erlt,
            erdp,
            gap,
    ):
        self.origin_time = origin_time
        self.time_std = time_std
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.nsta = nsta
        self.rms = rms
        self.erln = erln
        self.erlt = erlt
        self.erdp = erdp
        self.gap = gap

    def __repr__(self):
        return (
            f"Associated Event("
            f'{self.origin_time.isoformat("T")}, '
            f"lat: {self.latitude:.3f}, "
            f"lon: {self.longitude:.3f}, "
            f"dep: {self.depth:.3f}, "
            f"nsta: {self.nsta}, "
        )


class Client:
    """
    Client for sql database
    """

    def __init__(self, database, echo=False, build=False):
        self.database = database.split('.')[0]
        db_path = database

        if os.path.exists(db_path):
            self.engine = sqlalchemy.create_engine(
                f'sqlite:///{db_path}?check_same_thread=False', echo=echo)
        elif build:
            self.engine = sqlalchemy.create_engine(
                f'sqlite:///{db_path}?check_same_thread=False')
            self.con = self.engine.connect()
            self.con.execute(f"CREATE DATABASE {database}")  # create db
            self.con.execute(f"USE {database}")

        else:
            raise FileNotFoundError(f"{db_path} is not found")

        Base.metadata.create_all(bind=self.engine)
        self.Session = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = self.Session()

    def __repr__(self):
        return f'SQL Database({self.database})'

    def read_picks(self, picks, waveforms_dir):
        log = logging.getLogger(__name__)
        client = sds.Client(sds_root=waveforms_dir)
        fmtstr = os.path.join(
            "{year}",
            "{doy:03d}",
            "{station}",
            "{station}.{network}.{location}.{channel}.{year}.{doy:03d}",
        )
        # client.FMTSTR = fmtstr
        nslc = client.get_all_nslc()
        df_nslc = pd.DataFrame(nslc, columns=["net", "sta", "loc", "chan"])
        df_nslc = df_nslc.drop(columns="chan").drop_duplicates()
        df_nslc = df_nslc.set_index("sta")
        time1 = time.time()

        a = []
        bulk_picks = utils.parallel(
            picks, func=self.pickassoc, df_nslc=df_nslc
        )
        for item in bulk_picks:
            a = a + item
        bulk_picks = a

        self.session.bulk_save_objects(bulk_picks)
        self.session.commit()
        try:
            Index("time", PicksAssoc.time).create(self.engine)
        except sqlalchemy.exc.OperationalError as err:
            log.warning(err.args[0])

        log.debug(time.time() - time1)

    @staticmethod
    def pickassoc(pick, df_nslc):
        try:
            if isinstance(df_nslc.loc[pick.station]["net"], pd.Series):
                df_nslc_net = df_nslc.loc[pick.station]["net"][0]
            else:
                df_nslc_net = df_nslc.loc[pick.station]["net"]
        except:
            df_nslc_net = None
        try:
            if isinstance(df_nslc.loc[pick.station]["loc"], pd.Series):
                df_nslc_loc = df_nslc.loc[pick.station]["loc"][0]
            else:
                df_nslc_loc = df_nslc.loc[pick.station]["loc"]
        except:
            df_nslc_loc = None
        item = PicksAssoc(
            pick.station,
            df_nslc_net,
            df_nslc_loc,
            pick.time,
            pick.snr,
            trace_id=pick.trace_id,
            phase=pick.phase,
        )

        return item


if __name__ == "__main__":
    pass