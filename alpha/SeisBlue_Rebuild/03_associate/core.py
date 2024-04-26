# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from datetime import datetime
from typing import List
import sqlalchemy.ext.declarative
from sqlalchemy.dialects.mysql import DATETIME, DATE


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