"""
SQL Database
"""

import os
import operator
import contextlib
import numpy as np
import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.ext.declarative
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import DateTime
from obspy import UTCDateTime
from tqdm import tqdm

import obspy
import seisblue
import seisblue.core
import seisblue.io
import seisblue.plot
import seisblue.utils
from seisblue.associator.sql import AssociatedEvent, PicksAssoc, Candidate

Base = sqlalchemy.ext.declarative.declarative_base()


class Inventory(Base):
    """
    Inventory table for sql database.
    """
    __tablename__ = 'inventory'
    network = sqlalchemy.Column(sqlalchemy.String(6), nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6), primary_key=True)
    latitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    longitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    elevation = sqlalchemy.Column(sqlalchemy.Float, nullable=False)

    def __init__(self, net, sta, loc):
        self.network = net
        self.station = sta
        self.latitude = loc['latitude']
        self.longitude = loc['longitude']
        self.elevation = loc['elevation']

    def __repr__(self):
        return f"Inventory(" \
               f"Network={self.network}, " \
               f"Station={self.station}, " \
               f"Latitude={self.latitude:>7.4f}, " \
               f"Longitude={self.longitude:>8.4f}, " \
               f"Elevation={self.elevation:>6.1f})"

    def add_db(self, session):
        """
        Add data into session.

        :param session: SQL session.
        """
        session.add(self)


class Event(Base):
    """
    Event table for sql database.
    """
    __tablename__ = 'event'
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True)
    time = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    latitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    longitude = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    depth = sqlalchemy.Column(sqlalchemy.Float, nullable=False)

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
        """
        Add data into session.

        :type session: sqlalchemy.orm.session.Session
        :param session: SQL session.
        """
        session.add(self)


class Pick(Base):
    """
    Pick table for sql database.
    """
    __tablename__ = 'pick'
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True)
    time = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6),
                                                                nullable=False)
    phase = sqlalchemy.Column(sqlalchemy.String(90), nullable=False)
    tag = sqlalchemy.Column(sqlalchemy.String(10), nullable=False)
    snr = sqlalchemy.Column(sqlalchemy.Float)
    trace_id = sqlalchemy.Column(sqlalchemy.String(20))

    confidence = sqlalchemy.Column(sqlalchemy.Float)

    @compiles(DateTime, "mysql")
    def compile_datetime_mysql(type_, compiler, **kw):
        return "DATETIME(6)"

    def __init__(self, time, station, phase, tag, snr=None, confidence=None,trace_id = None):
        self.time = time
        self.station = station
        self.phase = phase
        self.tag = tag
        self.snr = snr
        self.confidence = confidence
        self.trace_id = trace_id

    def __repr__(self):
        return f"Pick(" \
               f"Time={self.time}, " \
               f"Station={self.station}, " \
               f"Phase={self.phase}, " \
               f"Tag={self.tag}, " \
               f"snr={self.snr}, " \
               f"confidence={self.confidence}," \
               f" trace_id={self.trace_id})"

    def add_db(self, session):
        """
        Add data into session.

        :type session: sqlalchemy.orm.session.Session
        :param session: SQL session.
        """
        session.add(self)


class TFRecord(Base):
    """
    TFRecord table for sql database.
    """
    __tablename__ = 'tfrecord'
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String(90), nullable=False)

    network = sqlalchemy.Column(sqlalchemy.String(6))
    station = sqlalchemy.Column(sqlalchemy.String(6),
                                sqlalchemy.ForeignKey('inventory.station'),
                                nullable=False)

    date = sqlalchemy.Column(sqlalchemy.Date, nullable=False)

    count = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    path = sqlalchemy.Column(sqlalchemy.String(180), nullable=False)
    tag = sqlalchemy.Column(sqlalchemy.String(10))

    def __init__(self, path, count):
        self.name = os.path.basename(path)
        network, station, location, channel, year, julday, suffix = \
            self.name.split('.')
        self.network = network
        self.station = station
        self.date = UTCDateTime(year=int(year), julday=int(julday)).datetime
        self.path = path
        self.count = count

    def __repr__(self):
        return f"TFRecord(" \
               f"Name={self.name}, " \
               f"Network={self.network}, " \
               f"Station={self.station}, " \
               f"Date={self.date}, " \
               f"Count={self.count}, " \
               f"Tag={self.tag})"

    def add_db(self, session):
        """
        Add data into session.

        :type session: sqlalchemy.orm.session.Session
        :param session: SQL session.
        """
        session.add(self)


class Waveform(Base):
    """
    Waveform table for sql database.
    """
    __tablename__ = 'waveform'
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True)
    starttime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    endtime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String(6),
                                sqlalchemy.ForeignKey('inventory.station'),
                                nullable=False)
    channel = sqlalchemy.Column(sqlalchemy.String(20), nullable=False)
    tfrecord = sqlalchemy.Column(sqlalchemy.String(180),
                                 # sqlalchemy.ForeignKey('tfrecord.id'),
                                 nullable=False)
    data_index = sqlalchemy.Column(sqlalchemy.Integer)

    def __init__(self, instance, tfrecord, data_index):
        self.starttime = UTCDateTime(instance.metadata.starttime).datetime
        self.endtime = UTCDateTime(instance.metadata.endtime).datetime
        self.station = instance.metadata.station
        self.channel = ', '.join(instance.trace.channel)
        self.tfrecord = tfrecord
        self.data_index = data_index

    def __repr__(self):
        return f"Waveform(" \
               f"Start={self.starttime}, " \
               f"End={self.endtime}, " \
               f"Station={self.station}, " \
               f"Channel={self.channel}, " \
               f"TFRecord={self.tfrecord}, " \
               f"Index={self.data_index})"

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
        config = seisblue.utils.Config()
        self.database = database
        db_path = os.path.join(config.sql_database, self.database)
        username = 'test'  # 資料庫帳號
        password = 'test123'  # 資料庫密碼
        host = '192.168.100.237'  # 資料庫位址
        port = '3306'

        if os.path.exists(db_path):

            self.engine = sqlalchemy.create_engine(
                f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}',
                echo=echo, pool_size=300, max_overflow=-1)
        elif build:
            self.engine = sqlalchemy.create_engine(
                f'mysql+pymysql://{username}:{password}@{host}:{port}',
                echo=echo)
            self.engine.execute(f"CREATE DATABASE {database}")  # create db
            self.engine.execute(f"USE {database}")

        else:
            raise FileNotFoundError(f'{db_path} is not found')

        Base.metadata.create_all(bind=self.engine)
        self.session = sqlalchemy.orm.sessionmaker(bind=self.engine)

    def __repr__(self):
        return f'SQL Database({self.database})'

    @staticmethod
    def get_table_class(table):
        """
        Returns related class from dict.

        :param str table: Keywords: inventory, event, pick, waveform.
        :return: SQL table class.
        """
        table_dict = {
            'inventory': Inventory,
            'event': Event,
            'pick': Pick,
            'waveform': Waveform,
            'tfrecord': TFRecord,
        }
        try:
            table_class = table_dict.get(table)
            return table_class

        except KeyError:
            msg = 'Please select table: inventory, event, pick, waveform'
            raise KeyError(msg)

    def read_hyp(self, hyp, network):
        """
        Add geometry data from .HYP file.

        seisblue.io.read_hyp wrap up.

        :param str hyp: STATION0.HYP file.
        :param str network: Output network name.
        """

        geom = seisblue.io.read_hyp(hyp)
        for sta, stats in geom.items():
            stats['network'] = network
        self.add_inventory(geom)

    def read_GDMSstations(self, nsta):
        geom = seisblue.io.read_GDMSstations(nsta)
        self.add_inventory(geom)

    def read_nsta(self,nsta24):
        config = seisblue.utils.Config()
        hyp_file = os.path.join(config.geom, nsta24)
        geom = {}
        with open(hyp_file, 'r') as file:
            station_list = []
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()

                sta = line[0:4].strip()
                lon = float(line[5:13].strip())
                lat = float(line[14:21].strip())
                elev = float(line[22:29].strip())
                loc = int(line[32:33].strip())
                source = line[34:38].strip()
                net = line[39:44].strip()
                equipments = line[45:48].strip()
                location = {'latitude': obspy.core.inventory.util.Latitude(lat),
                            'longitude': obspy.core.inventory.util.Longitude(lon),
                            'elevation': elev}

                geom[sta] = {'network': net,
                             'location': location}
        self.add_inventory(geom)
        # seisblue.io.write_hyp_station(geom,'/home/andy/Geom/STATION0.CWB.HYP')
        return station_list

    def read_kml_placemark(self, kml, network):
        """
        Add geometry data from .KML file.

        seisblue.io.read_kml_placemark wrap up.

        :param str kml: Google Earth .KML file.
        :param str network: Output network name.
        """

        geom = seisblue.io.read_kml_placemark(kml)
        for sta, stats in geom.items():
            stats['network'] = network
        self.add_inventory(geom)

    def add_inventory(self, geom):
        """
        Add geometry data from geometry dict.

        :param dict geom: Geometry dict.
        :param str network: Output network name.
        """
        with self.session_scope() as session:
            counter = 0
            for sta, stats in geom.items():
                Inventory(stats['network'], sta, stats['location']) \
                    .add_db(session)
                counter += 1

            print(f'Input {counter} stations')

    def get_inventories(self, station=None, network=None):
        """
        Returns query from inventory table.

        :param str station: Station name.
        :param str network: Network name.
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            query = session.query(Inventory)
            if station is not None:
                station = self.get_matched_list(
                    station, 'inventory', 'station')
                query = query.filter(Inventory.station.in_(station))
            if network is not None:
                network = self.get_matched_list(
                    network, 'inventory', 'network')
                query = query.filter(Inventory.network.in_(network))

        return query.all()

    def add_sfile_events(self, events, tag, remove_duplicates=True):
        """
        Add event data form catalog.

        :param events: events list.
        :param str tag: Pick tag.
        :param bool remove_duplicates: Removes duplicates in event table.
        """

        with self.session_scope() as session:
            event_count = 0
            pick_count = 0
            for event in events:
                if event.origins[0].latitude:
                    Event(event).add_db(session)
                    event_count += 1
                for pick in event.picks:
                    time = pick.time.datetime
                    station = pick.waveform_id.station_code
                    phase = pick.phase_hint
                    Pick(time, station, phase, tag).add_db(session)
                    pick_count += 1

            print(f'Input {event_count} events, {pick_count} picks')

        if remove_duplicates:
            self.remove_duplicates(
                'event',
                ['time', 'latitude', 'longitude', 'depth'])
            self.remove_duplicates(
                'pick',
                ['time', 'phase', 'station', 'tag'])
        self.session().commit()
    def get_events(self,
                   from_time=None, to_time=None,
                   west=None, east=None,
                   south=None, north=None,
                   from_depth=None, to_depth=None):
        """
        Returns query from event table.

        :param str from_time: From time.
        :param str to_time: To time.
        :param west: From West.
        :param east: To East.
        :param south: From South.
        :param north: To North.
        :param from_depth: From depth.
        :param to_depth: To depth.
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            query = session.query(Event)
            if from_time is not None:
                query = query.filter(Event.time >= from_time)
            if to_time is not None:
                query = query.filter(Event.time <= to_time)

            if west is not None:
                query = query.filter(Event.longitude >= west)
            if east is not None:
                query = query.filter(Event.latitude <= east)

            if south is not None:
                query = query.filter(Event.latitude >= south)
            if north is not None:
                query = query.filter(Event.latitude <= north)

            if from_depth is not None:
                query = query.filter(Event.depth >= from_depth)
            if to_depth is not None:
                query = query.filter(Event.depth <= to_depth)

        return query.all()

    def add_pick(self, time, station, phase, tag, snr, confidence=None,trace_id = None):
        with self.session_scope() as session:
            Pick(time, station, phase, tag, snr, confidence,trace_id).add_db(session)

    def get_picks(self,
                  from_time=None, to_time=None,
                  station=None, phase=None,
                  tag=None, low_snr=None, high_snr=None, low_confidence=None,high_confidence=None,
                  flatten=True):
        """
        Returns query from pick table.

        :param from_time: From time.
        :param to_time: To time.
        :param str station: Station name.
        :param str phase: Phase name.
        :param str tag: Catalog tag.
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            query = session.query(Pick)
            if from_time is not None:
                query = query.filter(Pick.time >= from_time)
            if to_time is not None:
                query = query.filter(Pick.time <= to_time)
            if station is not None:
                station = self.get_matched_list(station, 'pick', 'station')
                query = query.filter(Pick.station.in_(station))
            if phase is not None:
                phase = self.get_matched_list(phase, 'pick', 'phase')
                query = query.filter(Pick.phase.in_(phase))
            if tag is not None:
                tag = self.get_matched_list(tag, 'pick', 'tag')
                query = query.filter(Pick.tag.in_(tag))
            if low_snr is not None:
                query = query.filter(Pick.snr >= low_snr)
            if high_snr is not None:
                query = query.filter(Pick.snr <= high_snr)
            if low_confidence is not None:
                query = query.filter(Pick.confidence >= low_confidence)
            if high_confidence is not None:
                query = query.filter(Pick.confidence <= high_confidence)
        if flatten:
            return query.all()
        else:
            return query

    def read_tfrecord_header(self, tfr_list):
        """
        Sync header into SQL database from tfrecord dataset.

        :param tfr_list: TFRecord list.
        """
        try:
            for tfrecord in tqdm(tfr_list):
                dataset = seisblue.io.read_dataset(tfrecord)
                with self.session_scope() as session:
                    for index, example in enumerate(dataset):
                        instance = seisblue.core.Instance(example)
                        Waveform(instance, tfrecord, index).add_db(session)
                    TFRecord(tfrecord, index + 1).add_db(session)

        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        print(f'Input {index + 1} waveforms.')

    def get_tfrecord(self, network=None, station=None, path=None,
                     from_date=None, to_date=None, column=None):
        """
        Returns query from tfrecord table.

        :param str station: Station name.
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            if column is not None:
                query = session.query(getattr(TFRecord, column))
            else:
                query = session.query(TFRecord)

            if network is not None:
                network = self.get_matched_list(network, 'tfrecord', 'network')
                query = query.filter(TFRecord.network.in_(network))
            if station is not None:
                station = self.get_matched_list(station, 'tfrecord', 'station')
                query = query.filter(TFRecord.station.in_(station))
            if path is not None:
                path = self.get_matched_list(path, 'tfrecord', 'path')
                query = query.filter(TFRecord.path.in_(path))

            if from_date is not None:
                from_date = UTCDateTime(from_date).datetime
                query = query.filter(TFRecord.date >= from_date)
            if to_date is not None:
                to_date = UTCDateTime(to_date).datetime
                query = query.filter(TFRecord.date <= to_date)

        return query

    def get_waveform(self, from_time=None, to_time=None,
                     station=None, tfrecord=None):
        """
        Returns query from waveform table.

        .. note::
            If from_time or to_time is within the waveform, the waveform
            will be select.

        :param str from_time: Get which waveform endtime after from_time.
        :param str to_time: Get which waveform starttime after to_time.
        :param str/list station: Station name.
        :param str/list tfrecord: TFRecord path.
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """

        with self.session_scope() as session:
            query = session.query(Waveform)
            if from_time is not None:
                query = query.filter(Waveform.endtime >= from_time)
            if to_time is not None:
                query = query.filter(Waveform.starttime <= to_time)
            if station is not None:
                station = self.get_matched_list(
                    station, 'waveform', 'station')
                query = query.filter(Waveform.station.in_(station))
            if tfrecord is not None:
                tfrecord = self.get_matched_list(
                    tfrecord, 'waveform', 'tfrecord')
                query = query.filter(Waveform.tfrecord.in_(tfrecord))

        return query.all()

    def remove_duplicates(self, table, match_columns):
        """
        Removes duplicates data in given table.

        :param str table: Target table.
        :param list match_columns: List of column names.
            If all columns matches, then marks it as a duplicate data.
        """
        table = self.get_table_class(table)
        with self.session_scope() as session:
            attrs = operator.attrgetter(*match_columns)
            table_columns = attrs(table)
            distinct = session \
                .query(table, sqlalchemy.func.min(table.id)) \
                .group_by(*table_columns)
            distinct_id = distinct.with_entities(table.id).all()
            distinct_id = [int(np.array(i)) for i in distinct_id]
            duplicate = session \
                .query(table) \
                .filter(
                table.id.notin_(distinct_id)).delete(
                synchronize_session='fetch')
            print(f'Remove {duplicate} duplicate {table.__tablename__}s')

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
            query = session \
                .query(col(table).distinct()) \
                .order_by(col(table)) \
                .all()
        query = [q[0] for q in query]
        return query

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
        query = seisblue.utils.flatten_list(query)
        return query

    def clear_table(self, table):
        """
        Delete full table from database.

        :param table: Target table name.
        """
        table = self.get_table_class(table)
        with self.session_scope() as session:
            session.query(table).delete()

    @contextlib.contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        """
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception as exception:
            print(f'{exception.__class__.__name__}: {exception.__cause__}')
            session.rollback()
        finally:
            session.close()

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
                query = seisblue.utils.flatten_list(query)
                matched_list.extend(query)

        matched_list = sorted(list(set(matched_list)))
        return matched_list


class DatabaseInspector:
    """
    Main class for Database Inspector.
    """

    def __init__(self, database):
        if isinstance(database, str):
            try:
                database = seisblue.sql.Client(database)
            except Exception as exception:
                print(f'{exception.__class__.__name__}: {exception.__cause__}')

        self.database = database

    def inventory_summery(self):
        """
        Prints summery from geometry table.
        """
        stations = self.database.get_distinct_items('inventory', 'station')

        print('Station name:')
        print([station for station in stations], '\n')
        print(f'Total {len(stations)} stations\n')

        latitudes = self.database.get_distinct_items('inventory', 'latitude')
        longitudes = self.database.get_distinct_items('inventory', 'longitude')

        print('Station boundary:')
        print(f'West: {min(longitudes):>8.4f}')
        print(f'East: {max(longitudes):>8.4f}')
        print(f'South: {min(latitudes):>7.4f}')
        print(f'North: {max(latitudes):>7.4f}\n')

    def event_summery(self):
        """
        Prints summery from event table.
        """
        times = self.database.get_distinct_items('event', 'time')
        print('Event time duration:')
        print(f'From: {min(times).isoformat()}')
        print(f'To:   {max(times).isoformat()}\n')

        events = self.database.get_events()
        print(f'Total {len(events)} events\n')

        latitudes = self.database.get_distinct_items('event', 'latitude')
        longitudes = self.database.get_distinct_items('event', 'longitude')

        print('Station boundary:')
        print(f'West: {min(longitudes):>8.4f}')
        print(f'East: {max(longitudes):>8.4f}')
        print(f'South: {min(latitudes):>7.4f}')
        print(f'North: {max(latitudes):>7.4f}\n')

    def pick_summery(self):
        """
        Prints summery from pick table.
        """
        times = self.database.get_distinct_items('pick', 'time')
        print('Event time duration:')
        print(f'From: {min(times).isoformat()}')
        print(f'To:   {max(times).isoformat()}\n')

        print('Phase count:')
        phases = self.database.get_distinct_items('pick', 'phase')
        for phase in phases:
            picks = self.database.get_picks(phase=phase)
            print(f'{len(picks)} "{phase}" picks')
        print()

        pick_stations = self.database.get_distinct_items('pick', 'station')
        print(f'Picks cover {len(pick_stations)} stations:')
        print([station for station in pick_stations], '\n')

        no_pick_station = self.database.get_exclude_items(
            'inventory', 'station', pick_stations)
        if no_pick_station:
            print(f'{len(no_pick_station)} stations without picks:')
            print([station for station in no_pick_station], '\n')

        inventory_station = self.database \
            .get_distinct_items('inventory', 'station')
        no_inventory_station = self.database \
            .get_exclude_items('pick', 'station', inventory_station)

        if no_inventory_station:
            print(f'{len(no_inventory_station)} stations without geometry:')
            print([station for station in no_inventory_station], '\n')

    def waveform_summery(self):
        """
        Prints summery from waveform table.
        """
        starttimes = self.database.get_distinct_items('waveform', 'starttime')
        endtimes = self.database.get_distinct_items('waveform', 'endtime')
        print('Waveform time duration:')
        print(f'From: {min(starttimes).isoformat()}')
        print(f'To:   {max(endtimes).isoformat()}\n')

        waveforms = self.database.get_events()
        print(f'Total {len(waveforms)} events\n')

        stations = self.database.get_distinct_items('waveform', 'station')
        print(f'Picks cover {len(stations)} stations:')
        print([station for station in stations], '\n')

    def plot_map(self, **kwargs):
        """
        Plots station and event map.
        """
        inventories = self.database.get_inventories()
        events = self.database.get_events()
        seisblue.plot.plot_map(inventories, events, **kwargs)


def get_associates(database,
                   from_time=None,
                   to_time=None,
                   min_longitude=None,
                   max_longitude=None,
                   min_latitude=None,
                   max_latitude=None,
                   min_depth=None,
                   max_depth=None,
                   min_erlim=None,
                   max_erlim=None,
                   min_sta=None,
                   max_sta=None,
                   max_ot_uncert=None,
                   gap=None):
    db = seisblue.sql.Client(database)
    with db.session_scope() as session:
        query = session.query(AssociatedEvent)
        if from_time is not None:
            query = query.filter(AssociatedEvent.origin_time >= from_time)
        if to_time is not None:
            query = query.filter(AssociatedEvent.origin_time <= to_time)
        if min_longitude is not None:
            query = query.filter(AssociatedEvent.longitude >= min_longitude)
        if max_longitude is not None:
            query = query.filter(AssociatedEvent.longitude <= max_longitude)
        if min_latitude is not None:
            query = query.filter(AssociatedEvent.latitude >= min_latitude)
        if max_latitude is not None:
            query = query.filter(AssociatedEvent.latitude <= max_latitude)
        if min_depth is not None:
            query = query.filter(AssociatedEvent.depth >= min_depth)
        if max_depth is not None:
            query = query.filter(AssociatedEvent.depth <= max_depth)
        if min_erlim is not None:
            query = query.filter(
                (AssociatedEvent.erlt > min_erlim) |
                (AssociatedEvent.erln > min_erlim) |
                (AssociatedEvent.erdp > min_erlim))
        if max_erlim is not None:
            query = query.filter(AssociatedEvent.erlt <= max_erlim)
            query = query.filter(AssociatedEvent.erln <= max_erlim)
            query = query.filter(AssociatedEvent.erdp <= max_erlim)
        if min_sta is not None:
            query = query.filter(AssociatedEvent.nsta >= min_sta)
        if max_sta is not None:
            query = query.filter(AssociatedEvent.nsta <= max_sta)
        if max_ot_uncert is not None:
            query = query.filter(AssociatedEvent.time_std <= max_ot_uncert)
        if gap is not None:
            query = query.filter(AssociatedEvent.gap <= gap)

    return query.all()


def get_candidate(database, assoc_id=None):
    db = seisblue.sql.Client(database)
    with db.session_scope() as session:
        query = session.query(Candidate)
        if id is not None:
            query = query.filter(Candidate.assoc_id == assoc_id)

    return query.all()


def update_associate(database, assoc_id, update_dic):
    db = seisblue.sql.Client(database)
    with db.session_scope() as session:
        session.query(AssociatedEvent). \
            filter(AssociatedEvent.id == assoc_id).update(update_dic)


def pick_assoc_id(database, assoc_id):
    db = seisblue.sql.Client(database)
    with db.session_scope() as session:
        query = session.query(PicksAssoc).filter(
            PicksAssoc.assoc_id == assoc_id)
    return query.all()


if __name__ == "__main__":
    pass
