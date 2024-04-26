from dataclasses import dataclass, field, asdict
from obspy import UTCDateTime
import contextlib
import sqlalchemy
import sqlalchemy.orm
from typing import Optional
import operator
import logging
import os.path
import numpy as np
from sqlalchemy import and_
from typing import List
import scipy
from . import plot

Base = sqlalchemy.ext.declarative.declarative_base()


@dataclass
class TimeWindow:
    starttime: Optional[UTCDateTime] = None
    endtime: Optional[UTCDateTime] = None
    npts: Optional[int] = None
    sampling_rate: Optional[float] = None
    delta: Optional[float] = None
    station: str = field(repr=False, default=None)


@dataclass
class Inventory:
    network: Optional[str] = None
    station: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    time_window: Optional[TimeWindow] = None
    header: dict = field(repr=False, default=None)


class InventorySQL(Base):
    """
    Inventory table for sql database.
    """
    __tablename__ = 'inventory'
    id = sqlalchemy.Column(sqlalchemy.BigInteger()
                           .with_variant(sqlalchemy.Integer, "sqlite"),
                           primary_key=True)
    network = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    latitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    longitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    elevation = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    starttime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)
    endtime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)

    def __init__(self, inventory):
        self.network = inventory.network
        self.station = inventory.station
        self.latitude = inventory.latitude
        self.longitude = inventory.longitude
        self.elevation = inventory.elevation
        if inventory.time_window.starttime:
            self.starttime = inventory.time_window.starttime.datetime
            self.endtime = inventory.time_window.endtime.datetime
        else:
            self.starttime = inventory.time_window.starttime
            self.endtime = inventory.time_window.endtime

    def add_db(self, session):
        """
        Add data into session.

        :type session: sqlalchemy.orm.session.Session
        :param session: SQL session.
        """
        session.add(self)

    def transfer_to_dataclass(self):
        time_window = TimeWindow(
            starttime=UTCDateTime(self.starttime),
            endtime=UTCDateTime(self.endtime),
        )
        inventory = Inventory(
            network=self.network,
            station=self.station,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            time_window=time_window,
        )
        return inventory


@dataclass
class Event:
    origin_time: Optional[UTCDateTime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    header: dict = field(repr=False, default=None)


@dataclass
class EventSQL(Base):
    """
    Event table for sql database.
    """
    __tablename__ = 'event'
    id = sqlalchemy.Column(sqlalchemy.BigInteger()
                           .with_variant(sqlalchemy.Integer, "sqlite"),
                           primary_key=True)
    origin_time = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    latitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    longitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    depth = sqlalchemy.Column(sqlalchemy.Float, nullable=False)

    def __init__(self, event):
        self.origin_time = event.origin_time.datetime
        self.latitude = event.latitude
        self.longitude = event.longitude
        self.depth = event.depth

    def add_db(self, session):
        """
        Add data into session.

        :type session: sqlalchemy.orm.session.Session
        :param session: SQL session.
        """
        session.add(self)


@dataclass
class Pick:
    time: Optional[UTCDateTime] = None
    inventory: Optional[Inventory] = None
    phase: Optional[str] = None
    tag: Optional[str] = None
    snr: Optional[float] = None
    trace_id: Optional[str] = None
    confidence: Optional[float] = None
    header: dict = field(repr=False, default=None)


@dataclass
class PickSQL(Base):
    """
    Pick table for sql database.
    """
    __tablename__ = 'pick'
    id = sqlalchemy.Column(sqlalchemy.BigInteger()
                           .with_variant(sqlalchemy.Integer, "sqlite"),
                           primary_key=True)
    time = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6),
                                sqlalchemy.ForeignKey('inventory.station'),
                                nullable=False)
    phase = sqlalchemy.Column(sqlalchemy.String(90), nullable=False)
    tag = sqlalchemy.Column(sqlalchemy.String(10), nullable=False)
    snr = sqlalchemy.Column(sqlalchemy.Float)
    trace_id = sqlalchemy.Column(sqlalchemy.String(20))
    confidence = sqlalchemy.Column(sqlalchemy.Float)

    def __init__(self, pick):
        self.time = pick.time.datetime
        self.station = pick.inventory.station
        self.phase = pick.phase
        self.tag = pick.tag
        self.snr = None
        self.trace_id = None
        self.confidence = None

    def add_db(self, session):
        """
        Add data into session.

        :type session: sqlalchemy.orm.session.Session
        :param session: SQL session.
        """
        session.add(self)

    def transfer_to_dataclass(self):
        pick = Pick(
            time=UTCDateTime(self.time),
            phase=self.phase,
            tag=self.tag,
            inventory=Inventory(station=self.station),
            snr=self.snr,
            trace_id=self.trace_id,
            confidence=self.confidence
        )
        return pick


@dataclass
class Trace:
    inventory: Optional[Inventory] = None
    time_window: Optional[TimeWindow] = None
    channel: Optional[str] = None
    data: Optional[np.ndarray] = None
    header: dict = field(repr=False, default=None)


@dataclass
class Label:
    inventory: Optional[Inventory] = None
    time_window: Optional[TimeWindow] = None
    phase: Optional[List[str]] = None
    tag: Optional[str] = None
    data: Optional[np.ndarray] = None
    picks: List[Pick] = field(default_factory=list)

    def __post_init__(self):
        if self.time_window and self.phase:
            self.data = np.zeros([self.time_window.npts, len(self.phase)])

    def generate_pick_uncertainty_label(self, database, shape, half_width):
        """
        Add generated label to stream.

        :param str database:
        :param str shape: Label shape, see scipy.signal.windows.get_window().
        :param int half_width: Label half width in data point.
        :return:
        """
        db = Client(database=database)
        ph_index = {}
        for i, phase in enumerate(self.phase):
            ph_index[phase] = i

            picks_in_stream = db.get_picks(
                from_time=self.time_window.starttime.datetime,
                to_time=self.time_window.endtime.datetime,
                station=self.inventory.station,
                phase=phase,
                tag=self.tag)

            for pick in picks_in_stream:
                pick_time = UTCDateTime(pick.time) - self.time_window.starttime
                pick_time_index = int(pick_time / self.time_window.delta)
                self.data[pick_time_index, i] = 1
                self.picks.append(pick.transfer_to_dataclass())
            picks_time = self.data.copy()
            wavelet = scipy.signal.windows.get_window(shape, 2 * half_width)
            self.data[:, i] = scipy.signal.convolve(self.data[:, i],
                                                    wavelet[1:], mode="same")

        if 'E' in self.phase:
            eq_time = (picks_time[:, ph_index["P"]] - picks_time[:,
                                                      ph_index["S"]])
            eq_time = np.cumsum(eq_time)
            if np.any(eq_time < 0):
                eq_time += 1
            self.data[:, ph_index["E"]] = eq_time

        if 'N' in self.phase:
            # Make Noise window by 1 - P - S
            self.data[:, ph_index["N"]] = 1
            self.data[:, ph_index["N"]] -= self.data[:, ph_index["P"]]
            self.data[:, ph_index["N"]] -= self.data[:, ph_index["S"]]

        return self


@dataclass
class Instance:
    inventory: Optional[Inventory] = None
    time_window: Optional[TimeWindow] = None
    traces: Optional[List[Trace]] = None
    labels: Optional[List[Label]] = None
    trace_id: Optional[str] = None

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def plot(self, **kwargs):
        plot.plot_dataset(self, **kwargs)


class WaveformSQL(Base):
    """
    Waveform table for sql database.
    """

    __tablename__ = "waveform"
    trace_id = sqlalchemy.Column(sqlalchemy.String(180), primary_key=True)
    starttime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    endtime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    station = sqlalchemy.Column(
        sqlalchemy.String(6), sqlalchemy.ForeignKey("inventory.station"),
        nullable=False
    )
    network = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    channel = sqlalchemy.Column(sqlalchemy.String(20), nullable=False)
    dataset = sqlalchemy.Column(sqlalchemy.String(20), nullable=False)
    dataset_path = sqlalchemy.Column(sqlalchemy.String(180), nullable=False)

    def __init__(self, instance, dataset_path):
        self.starttime = instance.time_window.starttime.datetime
        self.endtime = instance.time_window.endtime.datetime
        self.station = instance.inventory.station
        self.network = instance.inventory.network
        self.channel = ", ".join([tr.channel for tr in instance.traces])
        self.dataset = dataset_path.split('/')[-1].split('.')[0]
        self.dataset_path = dataset_path
        self.trace_id = instance.trace_id

    def __repr__(self):
        return (
            f"Waveform("
            f"Start={self.starttime}, "
            f"End={self.endtime}, "
            f"Station={self.station}, "
            f"Network={self.network}, "
            f"Channel={self.channel}, "
            f"Dataset={self.dataset}, "
            f"Dataset_path={self.dataset_path}, "
            f"Trace_id={self.trace_id})"
        )

    def add_db(self, session):
        """
        Add data into session.

        :type session: sqlalchemy.orm.session.Session
        :param session: SQL session.
        """
        session.add(self)


class Client:
    """
    Client for sql database
    """

    def __init__(self, database, echo=False, build=False):
        self.database = database.split('.')[0]
        db_path = database

        if os.path.exists(db_path):
            self.engine = sqlalchemy.create_engine(
                f'sqlite:///{db_path}', echo=echo)
        elif build:
            self.engine = sqlalchemy.create_engine(
                f'sqlite:///{db_path}', echo=echo)
            self.con = self.engine.connect()
            self.con.execute(f"CREATE DATABASE {database}")  # create db
            self.con.execute(f"USE {database}")

        else:
            raise FileNotFoundError(f"{db_path} is not found")

        Base.metadata.create_all(bind=self.engine)
        self.session = sqlalchemy.orm.sessionmaker(bind=self.engine)

    def __repr__(self):
        return f'SQL Database({self.database})'

    def add_inventory(self, geoms, remove_duplicates=True):
        """
        Add geometry data into SQLDatabase.

        :param list geoms: List of inventory.
        :param bool remove_duplicates: Removes duplicates in inventory table.
        """
        log = logging.getLogger(__name__)
        with self.session_scope() as session:
            for geom in geoms:
                sql_item = InventorySQL(geom)
                sql_item.add_db(session)
            log.debug(
                f'Add {len(geoms)} inventories into database ({self.database}).')

        if remove_duplicates:
            self.remove_duplicates(
                'inventory',
                ['network', 'station', 'latitude', 'longitude', 'elevation'])

    def add_events(self, events, remove_duplicates=True):
        """
        Add event data into SQLDatabase.

        :param list events: List of event.
        :param bool remove_duplicates: Removes duplicates in event table.
        """
        log = logging.getLogger(__name__)
        with self.session_scope() as session:
            for event in events:
                sql_item = EventSQL(event)
                sql_item.add_db(session)
            log.debug(
                f'Add {len(events)} events into database ({self.database}).')
        if remove_duplicates:
            self.remove_duplicates(
                'event',
                ['origin_time', 'latitude', 'longitude', 'depth'])

    def add_picks(self, picks, remove_duplicates=True):
        """
        Add pick data into SQLDatabase.

        :param list picks: List of pick.
        :param bool remove_duplicates: Removes duplicates in pick table.
        """
        log = logging.getLogger(__name__)
        with self.session_scope() as session:
            for pick in picks:
                sql_item = PickSQL(pick)
                sql_item.add_db(session)
            log.debug(
                f'Add {len(picks)} picks into database ({self.database}).')
        if remove_duplicates:
            self.remove_duplicates(
                'pick',
                ['time', 'station', 'phase', 'tag'])

    def add_waveforms(self, instances, dataset_path, remove_duplicates=True):
        """
        Add waveform (path of dataset) into SQLDatabase.

        :param list[core.Instance] instances: List of instances.
        :param str dataset_path: hdf5 filepath
        :param bool remove_duplicates: Removes duplicates in waveform table.
        """
        log = logging.getLogger(__name__)

        with self.session_scope() as session:
            for instance in instances:
                sql_item = WaveformSQL(instance, dataset_path)
                sql_item.add_db(session)
            log.debug(
                f'Add {len(instances)} waveforms into database ({self.database}).')

        if remove_duplicates:
            self.remove_duplicates(
                'waveform', ['trace_id', 'tag'])

    @contextlib.contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        """
        log = logging.getLogger(__name__)
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception as exception:
            log.error(f'{exception.__class__.__name__}: {exception.__cause__}')
            session.rollback()
        finally:
            session.close()

    def remove_duplicates(self, table, match_columns):
        """
        Removes duplicates data in given table.

        :param str table: Target table.
        :param list match_columns: List of column names.
            If all columns matches, then marks it as a duplicate data.
        """
        log = logging.getLogger(__name__)
        table = self.get_table_class(table)
        with self.session_scope() as session:
            attrs = operator.attrgetter(*match_columns)
            table_columns = attrs(table)
            distinct = session \
                .query(table, sqlalchemy.func.min(table.id)) \
                .group_by(*table_columns)
            duplicate = session \
                .query(table) \
                .filter(table.id.notin_(distinct.with_entities(table.id))) \
                .delete(synchronize_session='fetch')
            if duplicate:
                log.debug(
                    f'Remove {duplicate} duplicate {table.__tablename__}s')

    @staticmethod
    def get_table_class(table):
        """
        Returns related class from dict.

        :param str table: Keywords: inventory, event, pick, waveform.
        :return: SQL table class.
        """
        table_dict = {
            'inventory': InventorySQL,
            'event': EventSQL,
            'pick': PickSQL,
            'waveform': WaveformSQL,
        }
        try:
            table_class = table_dict.get(table)
            return table_class

        except KeyError:
            msg = 'Please select table: inventory, event, pick, waveform'
            raise KeyError(msg)

    def get_distinct_items(self, table, column):
        """
        Returns a query of unique items.

        :param str table: Target table name.
        :param str column: Target column name.
        :rtype: list
        :return: A list of query.
        """
        table = self.get_table_class(table)
        col = operator.attrgetter(column)
        with self.session_scope() as session:
            pass
            query = session \
                .query(col(table).distinct()) \
                .order_by(col(table)) \
                .all()
        query = [q[0] for q in query]
        return query

    def get_items(self, table, column):
        """
        Returns a query of unique items.

        :param str table: Target table name.
        :param str column: Target column name.
        :rtype: list
        :return: A list of query.
        """
        table = self.get_table_class(table)
        col = operator.attrgetter(column)
        with self.session_scope() as session:
            pass
            query = session \
                .query(col(table)) \
                .order_by(col(table)) \
                .all()
        return query

    @staticmethod
    def replace_sql_wildcard(string):
        """
        Replaces posix wildcard characters to SQL wildcard characters.

        :param str string: Target string.
        :rtype: str
        :return: Replaced string.
        """
        string = string.replace('?', '_')
        string = string.replace('*', '%')
        return string

    def get_matched_list(self, wildcard_list, table, column):
        """
        Gets wildcard match list in column.

        :param str/list wildcard_list:
        :param str table: Table name.
        :param str column: Column name.
        :return:
        """
        if isinstance(wildcard_list, str):
            wildcard_list = [wildcard_list]

        table = self.get_table_class(table)

        matched_list = []
        for wildcard in wildcard_list:
            wildcard = self.replace_sql_wildcard(wildcard)
            with self.session_scope() as session:
                query = session.query(getattr(table, column)) \
                    .filter(getattr(table, column).like(wildcard)) \
                    .distinct() \
                    .order_by(getattr(table, column)) \
                    .all()

                query = [item for sublist in query for item in sublist]
                matched_list.extend(query)

        matched_list = sorted(list(set(matched_list)))
        return matched_list

    def get_exclude_items(self, table, column, exclude):
        """
        Returns a query of exclude items.

        :param str table: Target table name.
        :param str column: Target column name.
        :param exclude: Exclude list.
        :rtype: list
        :return: A list of query.
        """
        table = self.get_table_class(table)
        with self.session_scope() as session:
            col = operator.attrgetter(column)
            query = session \
                .query(col(table)) \
                .order_by(col(table)) \
                .filter(col(table).notin_(exclude)) \
                .all()
        query = [item for sublist in query for item in sublist]
        return query

    def get_picks(
            self,
            from_time=None,
            to_time=None,
            station=None,
            phase=None,
            tag=None,
            low_snr=None,
            high_snr=None,
            low_confidence=None,
            high_confidence=None,
            flatten=True,
    ):
        """
        Returns query from pick table.

        :param from_time: From time.
        :param to_time: To time.
        :param str station: Station name.
        :param str phase: Phase name.
        :param str tag: Catalog tag.
        :param: int low_snr:
        :param: int high_snr:
        :param: int low_confidence:
        :param: int high_confidence:
        :param: bool flatten:
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            query = session.query(PickSQL)
            if from_time is not None:
                query = query.filter(PickSQL.time >= from_time)
            if to_time is not None:
                query = query.filter(PickSQL.time <= to_time)
            if station is not None:
                station = self.get_matched_list(station, "pick", "station")
                query = query.filter(PickSQL.station.in_(station))
            if phase is not None:
                phase = self.get_matched_list(phase, "pick", "phase")
                query = query.filter(PickSQL.phase.in_(phase))
            if tag is not None:
                tag = self.get_matched_list(tag, "pick", "tag")
                query = query.filter(PickSQL.tag.in_(tag))
            if low_snr is not None:
                query = query.filter(PickSQL.snr >= low_snr)
            if high_snr is not None:
                query = query.filter(PickSQL.snr <= high_snr)
            if low_confidence is not None:
                query = query.filter(PickSQL.confidence >= low_confidence)
            if high_confidence is not None:
                query = query.filter(PickSQL.confidence <= high_confidence)
        if flatten:
            return query.all()
        else:
            return query

    def get_inventory(
            self,
            network=None,
            station=None,
            time=None,
            flatten=True,
    ):
        """
        Returns query from inventory table.

        :param str network: Network name.
        :param str station: Station name.
        :param: UTCDateTime time:
        :param: bool flatten:
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            query = session.query(InventorySQL)
            if network is not None:
                network = self.get_matched_list(network, "inventory", "network")
                query = query.filter(InventorySQL.network.in_(network))
            if station is not None:
                station = self.get_matched_list(station, "inventory", "station")
                query = query.filter(InventorySQL.station.in_(station))
            if time is not None:
                query = query.filter(
                    and_(
                        time.datetime >= InventorySQL.starttime,
                        time.datetime <= InventorySQL.endtime
                    )
                )
        if flatten:
            return query.all()
        else:
            return query

    def get_dataset(
            self,
            network=None,
            station=None,
            dataset=None,
            from_date=None,
            to_date=None,
            column=None,
            flatten=True,
    ):
        """
        Returns query from waveform table.

        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            if column is not None:
                query = session.query(getattr(WaveformSQL, column))
            else:
                query = session.query(WaveformSQL)

            if network is not None:
                network = self.get_matched_list(network, "waveform", "network")
                query = query.filter(WaveformSQL.network.in_(network))
            if station is not None:
                station = self.get_matched_list(station, "waveform", "station")
                query = query.filter(WaveformSQL.station.in_(station))
            if dataset is not None:
                dataset = self.get_matched_list(dataset, "waveform", "dataset")
                query = query.filter(WaveformSQL.dataset.in_(dataset))
            if from_date is not None:
                from_date = UTCDateTime(from_date).datetime
                query = query.filter(WaveformSQL.starttime >= from_date)
            if to_date is not None:
                to_date = UTCDateTime(to_date).datetime
                query = query.filter(WaveformSQL.starttime <= to_date)

        if flatten:
            return query.all()
        else:
            return query


if __name__ == '__main__':
    pass
