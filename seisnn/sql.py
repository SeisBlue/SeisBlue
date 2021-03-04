"""
SQL Database
"""

import os
import operator
import contextlib

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.ext.declarative
from obspy import UTCDateTime

import seisnn.core
import seisnn.io
import seisnn.plot
import seisnn.processing
import seisnn.utils

Base = sqlalchemy.ext.declarative.declarative_base()


class Inventory(Base):
    """
    Inventory table for sql database.
    """
    __tablename__ = 'inventory'
    network = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String, primary_key=True)
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
    id = sqlalchemy.Column(sqlalchemy.BigInteger()
                           .with_variant(sqlalchemy.Integer, "sqlite"),
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
    id = sqlalchemy.Column(sqlalchemy.BigInteger()
                           .with_variant(sqlalchemy.Integer, "sqlite"),
                           primary_key=True)
    time = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String,
                                sqlalchemy.ForeignKey('inventory.station'),
                                nullable=False)
    phase = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    tag = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    split = sqlalchemy.Column(sqlalchemy.String)
    snr = sqlalchemy.Column(sqlalchemy.Float)

    def __init__(self, time, station, phase, tag):
        self.time = time
        self.station = station
        self.phase = phase
        self.tag = tag

    def __repr__(self):
        return f"Pick(" \
               f"Time={self.time}, " \
               f"Station={self.station}, " \
               f"Phase={self.phase}, " \
               f"Tag={self.tag}, " \
               f"Split={self.split}, " \
               f"SNR={self.snr})"

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
    id = sqlalchemy.Column(sqlalchemy.BigInteger()
                           .with_variant(sqlalchemy.Integer, "sqlite"),
                           primary_key=True)
    starttime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    endtime = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    station = sqlalchemy.Column(sqlalchemy.String,
                                sqlalchemy.ForeignKey('inventory.station'),
                                nullable=False)
    channel = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    dataset = sqlalchemy.Column(sqlalchemy.String)
    data_index = sqlalchemy.Column(sqlalchemy.Integer)

    def __init__(self, instance, dataset, data_index):
        self.starttime = UTCDateTime(instance.metadata.starttime).datetime
        self.endtime = UTCDateTime(instance.metadata.endtime).datetime
        self.station = instance.metadata.station
        self.channel = ', '.join(instance.trace.channel)
        self.dataset = dataset
        self.data_index = data_index

    def __repr__(self):
        return f"Waveform(" \
               f"Start={self.starttime}, " \
               f"End={self.endtime}, " \
               f"Station={self.station}, " \
               f"Channel={self.channel}, " \
               f"Dataset={self.dataset}, " \
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

    def __init__(self, database, echo=False):
        config = seisnn.utils.get_config()
        self.database = database
        db_path = os.path.join(config['SQL_ROOT'], self.database)
        self.engine = sqlalchemy.create_engine(
            f'sqlite:///{db_path}?check_same_thread=False',
            echo=echo)
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

        seisnn.io.read_hyp wrap up.

        :param str hyp: STATION0.HYP file.
        :param str network: Output network name.
        """

        geom = seisnn.io.read_hyp(hyp)
        self.add_inventory(geom, network)

    def read_kml_placemark(self, kml, network):
        """
        Add geometry data from .KML file.

        seisnn.io.read_kml_placemark wrap up.

        :param str kml: Google Earth .KML file.
        :param str network: Output network name.
        """

        geom = seisnn.io.read_kml_placemark(kml)
        self.add_inventory(geom, network)

    def add_inventory(self, geom, network):
        """
        Add geometry data from geometry dict.

        :param dict geom: Geometry dict.
        :param str network: Output network name.
        """
        with self.session_scope() as session:
            counter = 0
            for sta, loc in geom.items():
                Inventory(network, sta, loc).add_db(session)
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
            if station:
                station = self.replace_sql_wildcard(station)
                query = query.filter(Inventory.station.like(station))
            if network:
                network = self.replace_sql_wildcard(network)
                query = query.filter(Inventory.network.like(network))

        return query

    def inventory_summery(self):
        """
        Prints summery from geometry table.
        """
        stations = self.get_distinct_items('inventory', 'station')

        print('Station name:')
        print([station for station in stations], '\n')
        print(f'Total {len(stations)} stations\n')

        latitudes = self.get_distinct_items('inventory', 'latitude')
        longitudes = self.get_distinct_items('inventory', 'longitude')

        print('Station boundary:')
        print(f'West: {min(longitudes):>8.4f}')
        print(f'East: {max(longitudes):>8.4f}')
        print(f'South: {min(latitudes):>7.4f}')
        print(f'North: {max(latitudes):>7.4f}\n')

    def plot_map(self):
        """
        Plots station and event map.
        """
        inventories = self.get_inventories()
        events = self.get_events()

        seisnn.plot.plot_map(inventories, events)

    def add_events(self, catalog, tag, remove_duplicates=True):
        """
        Add event data form catalog.

        :param str catalog: Catalog name.
        :param str tag: Pick tag.
        :param bool remove_duplicates: Removes duplicates in event table.
        """

        events = seisnn.io.read_event_list(catalog)
        with self.session_scope() as session:
            event_count = 0
            pick_count = 0
            for event in events:
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

        return query

    def add_pick(self, time, station, phase, tag):
        with self.session_scope() as session:
            Pick(time, station, phase, tag).add_db(session)

    def get_picks(self, from_time=None, to_time=None,
                  station=None, phase=None, tag=None):
        """
        Returns query from pick table.

        :param str from_time: From time.
        :param str to_time: To time.
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
                station = self.replace_sql_wildcard(station)
                query = query.filter(Pick.station.like(station))
            if phase is not None:
                query = query.filter(Pick.phase.like(phase))
            if tag is not None:
                query = query.filter(Pick.tag.like(tag))

        return query

    def event_summery(self):
        """
        Prints summery from event table.
        """
        times = self.get_distinct_items('event', 'time')
        print('Event time duration:')
        print(f'From: {min(times).isoformat()}')
        print(f'To:   {max(times).isoformat()}\n')

        events = self.get_events().all()
        print(f'Total {len(events)} events\n')

        latitudes = self.get_distinct_items('event', 'latitude')
        longitudes = self.get_distinct_items('event', 'longitude')

        print('Station boundary:')
        print(f'West: {min(longitudes):>8.4f}')
        print(f'East: {max(longitudes):>8.4f}')
        print(f'South: {min(latitudes):>7.4f}')
        print(f'North: {max(latitudes):>7.4f}\n')

    def pick_summery(self):
        """
        Prints summery from pick table.
        """
        times = self.get_distinct_items('pick', 'time')
        print('Event time duration:')
        print(f'From: {min(times).isoformat()}')
        print(f'To:   {max(times).isoformat()}\n')

        print('Phase count:')
        phases = self.get_distinct_items('pick', 'phase')
        for phase in phases:
            picks = self.get_picks(phase=phase).all()
            print(f'{len(picks)} "{phase}" picks')
        print()

        pick_stations = self.get_distinct_items('pick', 'station')
        print(f'Picks cover {len(pick_stations)} stations:')
        print([station for station in pick_stations], '\n')

        no_pick_station = self.get_exclude_items('inventory', 'station',
                                                 pick_stations)
        if no_pick_station:
            print(f'{len(no_pick_station)} stations without picks:')
            print([station for station in no_pick_station], '\n')

        inventory_station = self.get_distinct_items('inventory', 'station')
        no_inventory_station = self.get_exclude_items('pick', 'station',
                                                      inventory_station)

        if no_inventory_station:
            print(f'{len(no_inventory_station)} stations without geometry:')
            print([station for station in no_inventory_station], '\n')

    def generate_training_data(self, pick_list, dataset, chunk_size=64):
        """
        Generate TFrecords from database.

        :param pick_list: List of picks from Pick SQL query.
        :param str dataset: Output directory name.
        :param int chunk_size: Number of data stores in TFRecord.
        """
        seisnn.processing.generate_training_data(
            pick_list, dataset, self.database, chunk_size)

    def read_tfrecord_header(self, dataset):
        """
        Sync header into SQL database from tfrecord dataset.

        :param str dataset: Dataset name.
        """
        ds = seisnn.io.read_dataset(dataset)
        index = 0
        with self.session_scope() as session:
            for example in ds:
                instance = seisnn.core.Instance(example)
                try:
                    Waveform(instance, dataset, index).add_db(session)
                    index += 1
                except Exception as error:
                    print(f'{type(error).__name__}: {error}')

        print(f'Input {index} waveforms.')

    def get_waveform(self, from_time=None, to_time=None, station=None):
        """
        Returns query from waveform table.

        .. note::
            If from_time or to_time is within the waveform, the waveform
            will be select.

        :param str from_time: Get which waveform endtime after from_time.
        :param str to_time: Get which waveform starttime after to_time.
        :param str station: Station name.
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        with self.session_scope() as session:
            query = session.query(Waveform)
            if from_time is not None:
                query = query.filter(from_time <= Waveform.endtime)
            if to_time is not None:
                query = query.filter(Waveform.starttime <= to_time)
            if station is not None:
                station = self.replace_sql_wildcard(station)
                query = query.filter(Waveform.station.like(station))

        return query

    def waveform_summery(self):
        """
        Prints summery from waveform table.
        """
        starttimes = self.get_distinct_items('waveform', 'starttime')
        endtimes = self.get_distinct_items('waveform', 'endtime')
        print('Waveform time duration:')
        print(f'From: {min(starttimes).isoformat()}')
        print(f'To:   {max(endtimes).isoformat()}\n')

        waveforms = self.get_events().all()
        print(f'Total {len(waveforms)} events\n')

        stations = self.get_distinct_items('waveform', 'station')
        print(f'Picks cover {len(stations)} stations:')
        print([station for station in stations], '\n')

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
            duplicate = session \
                .query(table) \
                .filter(table.id.notin_(distinct.with_entities(table.id))) \
                .delete(synchronize_session='fetch')
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
        query = [q[0] for q in query]
        return query

    def clear_table(self, table):
        table = self.get_table_class(table)
        with self.session_scope() as session:
            session.query(table).delete()

    def split_pick(self,proportion):
        with self.session_scope() as session:

            query = session.query(Pick).slice(0,20)
            query.update({Pick.split: "1"})

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


if __name__ == "__main__":
    pass
