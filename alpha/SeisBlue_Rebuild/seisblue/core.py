# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from datetime import datetime
from typing import List
import sqlalchemy.ext.declarative
from sqlalchemy.dialects.mysql import DATETIME, DATE


Base = sqlalchemy.ext.declarative.declarative_base()


@dataclass
class TimeWindow:
    starttime: Optional[Union[datetime, str]] = None
    endtime: Optional[Union[datetime, str]] = None
    npts: Optional[int] = None
    samplingrate: Optional[float] = None
    delta: Optional[float] = None
    network: Optional[str] = None
    station: Optional[str] = None


@dataclass
class Inventory:
    network: Optional[str] = None
    station: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    timewindow: Optional[TimeWindow] = None
    header: dict = field(repr=False, default=None)


@dataclass
class NodalPlane:
    strike: Optional[float] = None
    strike_errors: Optional[float] = None
    dip: Optional[float] = None
    dip_errors: Optional[float] = None
    rake: Optional[float] = None
    rake_errors: Optional[float] = None
    fault_uncertanity: Optional[float] = None
    auxiliary_uncertanity: Optional[float] = None
    method: Optional[str] = None


@dataclass
class Event:
    time: Optional[Union[datetime, str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    magnitude: Optional[float] = None
    nodal_plane: Optional[NodalPlane] = None
    header: dict = field(repr=False, default=None)


@dataclass
class Pick:
    time: Optional[Union[datetime, str]] = None
    inventory: Optional[Inventory] = None
    phase: Optional[str] = None
    tag: Optional[str] = None
    snr: Optional[float] = None
    traceid: Optional[str] = None
    confidence: Optional[float] = None
    polarity: Optional[str] = None
    azimuth: Optional[float] = None
    takeoff_angle: Optional[float] = None
    header: dict = field(repr=False, default=None)
    sfile: Optional[str] = None


@dataclass
class Stream:
    inventory: Optional[Inventory] = None
    timewindow: Optional[TimeWindow] = None
    channel: Optional[str] = None
    data: Optional[np.ndarray] = field(default=None)
    header: dict = field(repr=False, default=None)


@dataclass
class Label:
    inventory: Optional[Inventory] = None
    timewindow: Optional[TimeWindow] = None
    phase: Optional[str] = None
    tag: Optional[str] = None
    data: np.ndarray = field(default=None)
    picks: List[Pick] = field(default_factory=list)


@dataclass
class Instance:
    inventory: Optional[Inventory] = None
    timewindow: Optional[TimeWindow] = None
    features: Optional[Stream] = None
    labels: Optional[List[Label]] = field(default_factory=list)
    dataset: Optional[str] = None
    datasetpath: Optional[str] = None
    id: Optional[str] = None


class InventorySQL(Base):
    """
    Inventory table for sql database.
    """
    __tablename__ = 'inventory'
    id = sqlalchemy.Column(sqlalchemy.BigInteger(), primary_key=True)
    network = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    latitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    longitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    elevation = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    starttime = sqlalchemy.Column(DATE, nullable=True)
    endtime = sqlalchemy.Column(DATE, nullable=True)

    def __init__(self, inventory):
        attrs = ['network', 'station', 'latitude', 'longitude', 'elevation']
        for attr in attrs:
            setattr(self, attr, getattr(inventory, attr, None))

        if inventory.timewindow:
            for attr in ['starttime', 'endtime']:
                setattr(self, attr, getattr(inventory.timewindow, attr, None))

    def __repr__(self):
        attrs = [f"{attr}={getattr(self, attr) or 'None'}" for attr in
                 ['network', 'station', 'latitude', 'longitude', 'elevation', 'starttime', 'endtime', 'id']]
        return "Inventory(" + ", ".join(attrs) + ")"

    def to_dataclass(self):
        if self.starttime and self.endtime:
            timewindow = TimeWindow(
                starttime=datetime.fromisoformat(str(self.starttime)),
                endtime=datetime.fromisoformat(str(self.endtime)),
            )
        else:
            timewindow = None
        inventory = Inventory(
            network=self.network,
            station=self.station,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            timewindow=timewindow
        )

        return inventory


class EventSQL(Base):
    """
    Event table for sql database.
    """
    __tablename__ = 'event'
    id = sqlalchemy.Column(sqlalchemy.BigInteger, primary_key=True)
    time = sqlalchemy.Column(DATETIME(fsp=2), nullable=False, index=True)
    latitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    longitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    depth = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    magnitude = sqlalchemy.Column(sqlalchemy.Float)
    strike = sqlalchemy.Column(sqlalchemy.Float)
    dip = sqlalchemy.Column(sqlalchemy.Float)
    rake = sqlalchemy.Column(sqlalchemy.Float)

    def __init__(self, event):
        attrs = ['time', 'latitude', 'longitude', 'depth', 'magnitude']
        for attr in attrs:
            setattr(self, attr, getattr(event, attr, None))

        if event.nodal_plane:
            for attr in ['strike', 'dip', 'rake']:
                setattr(self, attr, getattr(event.nodal_plane, attr, None))

    def __repr__(self):
        attrs = [f"{attr}={getattr(self, attr) or 'None'}" for attr in
                 ['time', 'latitude', 'longitude', 'depth', 'magnitude', 'strike', 'dip', 'rake']]
        return "Event(" + ", ".join(attrs) + ")"


class PickSQL(Base):
    """
    Pick table for sql database.
    """
    __tablename__ = 'pick'
    id = sqlalchemy.Column(sqlalchemy.BigInteger(), primary_key=True)
    time = sqlalchemy.Column(DATETIME(fsp=2), nullable=False, index=True)
    network = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    phase = sqlalchemy.Column(sqlalchemy.String(90), nullable=False)
    tag = sqlalchemy.Column(sqlalchemy.String(10), nullable=False)
    snr = sqlalchemy.Column(sqlalchemy.Float)
    confidence = sqlalchemy.Column(sqlalchemy.Float)
    # polarity = sqlalchemy.Column(sqlalchemy.String(10), nullable=True)

    def __init__(self, pick):
        for attr in ['time', 'phase', 'tag', 'polarity', 'snr', 'confidence']:
            setattr(self, attr, getattr(pick, attr, None))
        for attr in ['station', 'network']:
            setattr(self, attr, getattr(pick.inventory, attr, None))

    def __repr__(self):
        attrs = [f"{attr}={getattr(self, attr) or 'None'}" for attr in
                 ['time', 'station', 'network', 'phase', 'tag', 'snr', 'confidence', 'polarity']]
        return "Pick(" + ", ".join(attrs) + ")"

    def to_dataclass(self):
        pick = Pick(
            time=datetime.fromisoformat(str(self.time)),
            phase=self.phase,
            tag=self.tag,
            inventory=Inventory(station=self.station, network=self.network),
            snr=self.snr,
            confidence=self.confidence
        )
        return pick


class WaveformSQL(Base):
    """
    Waveform table for sql database.
    """

    __tablename__ = "waveform"
    id = sqlalchemy.Column(sqlalchemy.BigInteger(), primary_key=True)
    starttime = sqlalchemy.Column(DATETIME(fsp=2), nullable=False, index=True)
    endtime = sqlalchemy.Column(DATETIME(fsp=2), nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    network = sqlalchemy.Column(sqlalchemy.String(6))
    channel = sqlalchemy.Column(sqlalchemy.String(40), nullable=False)
    dataset = sqlalchemy.Column(sqlalchemy.String(20), nullable=False)
    datasetpath = sqlalchemy.Column(sqlalchemy.String(180), nullable=False)

    def __init__(self, instance):
        self.starttime = instance.timewindow.starttime
        self.endtime = instance.timewindow.endtime
        self.station = instance.inventory.station
        self.network = instance.inventory.network
        self.channel = instance.features.channel
        self.dataset = instance.dataset
        self.datasetpath = instance.datasetpath

    def __repr__(self):
        attrs = [f"{attr}={getattr(self, attr) or 'None'}" for attr in
                 ['starttime', 'endtime', 'station', 'network', 'channel', 'dataset', 'datasetpath', 'id']]
        return "Waveform(" + ", ".join(attrs) + ")"


@dataclass
class PickAssoc:
    sta: Optional[str] = None
    net: Optional[str] = None
    loc: Optional[str] = None
    time: Optional[Union[datetime, str]] = None
    snr: Optional[float] = None
    phase: Optional[str] = None
    locate_flag: Optional[bool] = None
    assoc_id: Optional[int] = None
    trace_id: Optional[str] = None


class PickAssocSQL(Base):
    __tablename__ = "pick_assoc"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    sta = sqlalchemy.Column(sqlalchemy.String(5), nullable=False)
    net = sqlalchemy.Column(sqlalchemy.String(2), nullable=True)
    loc = sqlalchemy.Column(sqlalchemy.String(2), nullable=True)
    time = sqlalchemy.Column(DATETIME(fsp=2), nullable=False)
    snr = sqlalchemy.Column(sqlalchemy.Float)
    phase = sqlalchemy.Column(sqlalchemy.String(5), nullable=False)
    locate_flag = sqlalchemy.Column(sqlalchemy.Boolean)
    assoc_id = sqlalchemy.Column(sqlalchemy.Integer)
    trace_id = sqlalchemy.Column(sqlalchemy.String(20))

    def __init__(self, pick):
        self.sta = pick.sta
        self.net = "" if not pick.net else pick.net
        self.loc = "" if not pick.loc else pick.loc
        self.time = pick.time
        self.snr = pick.snr
        self.phase = pick.phase
        self.locate_flag = None
        self.assoc_id = None
        self.trace_id = pick.trace_id

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
    distance = sqlalchemy.Column(sqlalchemy.Float)

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
        self.distance = None

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
        pick_p = session.query(PickAssoc).filter(PickAssoc.id == self.p_id)
        for pick in pick_p:
            pick.phase = "P"
            pick.assoc_id = assoc_id
            pick.locate_flag = locate_flag

        pick_s = session.query(PickAssoc).filter(PickAssoc.id == self.s_id)
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



model_dict = {
    'inventory': Inventory,
    'timewindow': TimeWindow,
    'features': Stream,
    'pick': Pick,
    'picks': Pick,
    'instance': Instance,
}


if __name__ == '__main__':
    pass