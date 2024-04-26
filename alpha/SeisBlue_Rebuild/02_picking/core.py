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