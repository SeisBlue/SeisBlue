# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, Union
from datetime import datetime
import numpy as np
from typing import List


@dataclass
class StationTimeWindow:
    starttime: Optional[Union[datetime, str]] = None
    endtime: Optional[Union[datetime, str]] = None


@dataclass
class Inventory:
    network: Optional[str] = None
    station: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    timewindow: Optional[StationTimeWindow] = None
    header: dict = field(repr=False, default=None)


@dataclass
class Pick:
    pick_id: Optional[str] = None
    time: Optional[Union[datetime, str]] = None
    inventory: Optional[Inventory] = None
    phase: Optional[str] = None
    polarity: Optional[str] = None
    raw_polarity: Optional[str] = None
    tag: Optional[str] = None
    snr: Optional[float] = None
    traceid: Optional[str] = None
    confidence: Optional[float] = None
    azimuth: Optional[float] = None
    takeoff_angle: Optional[float] = None
    header: dict = field(repr=False, default=None)


@dataclass
class TimeWindow:
    starttime: Optional[Union[datetime, str]] = None
    endtime: Optional[Union[datetime, str]] = None
    npts: Optional[int] = None
    samplingrate: Optional[float] = None
    delta: Optional[float] = None
    inventory: Optional[Inventory] = None
    picks: Optional[List[Pick]] = field(default_factory=list)


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
class Kagan:
    np1: NodalPlane
    np2: NodalPlane
    kagan_angle: float

@dataclass
class Event:
    time: Optional[Union[datetime, str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    magnitude: Optional[float] = None
    focal_mechanism: Optional[NodalPlane] = None
    kagan: Optional[Kagan] = None
    header: dict = field(repr=False, default=None)


@dataclass
class Trace:
    inventory: Optional[Inventory] = None
    timewindow: Optional[TimeWindow] = None
    channel: Optional[str] = None
    data: Optional[np.ndarray] = field(default=None)
    header: dict = field(repr=False, default=None)

@dataclass
class Label:
    inventory: Optional[Inventory] = None
    timewindow: Optional[TimeWindow] = None
    tag: Optional[str] = None
    data: np.ndarray = field(default=None)
    pick: Optional[Pick] = None

@dataclass
class Instance:
    inventory: Optional[Inventory] = None
    timewindow: Optional[TimeWindow] = None
    traces: Optional[List[Trace]] = field(default_factory=list)
    labels: Optional[List[Label]] = field(default_factory=list)
    Spick: Optional[Pick] = None
    Ppick: Optional[Pick] = None
    id: Optional[str] = None
    dataset: Optional[str] = None
    datasetpath: Optional[str] = None

@dataclass
class EventInstance:
    instances: Optional[List[Instance]] = field(default_factory=list)
    event: Optional[Event] = None
    id: Optional[str] = None
    dataset: Optional[str] = None
    datasetpath: Optional[str] = None

