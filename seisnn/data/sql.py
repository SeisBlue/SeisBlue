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

import seisnn.processing
from seisnn.data import core, io
from seisnn import plot
from seisnn import utils

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
    dataset = sqlalchemy.Column(sqlalchemy.String)
    data_index = sqlalchemy.Column(sqlalchemy.Integer)

    def __init__(self, instance, dataset, data_index):
        self.starttime = UTCDateTime(instance.starttime).datetime
        self.endtime = UTCDateTime(instance.endtime).datetime
        self.station = instance.station
        self.dataset = dataset
        self.data_index = data_index

    def __repr__(self):
        return f"Waveform(" \
               f"Start={self.starttime}, " \
               f"End={self.endtime}, " \
               f"Station={self.station}, " \
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
        config = utils.get_config()
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

        geom = io.read_hyp(hyp)
        self.add_geom(geom, network)

    def read_kml_placemark(self, kml, network):
        """
        Add geometry data from .KML file.

        seisnn.io.read_kml_placemark wrap up.

        :param str kml: Google Earth .KML file.
        :param str network: Output network name.
        """

        geom = io.read_kml_placemark(kml)
        self.add_geom(geom, network)

    def add_geom(self, geom, network):
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

    def get_geom(self, station=None, network=None):
        """
        Returns query from geometry table.

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

    def geom_summery(self):
        """
        Prints summery from geometry table.
        """
        with self.session_scope() as session:
            station = session \
                .query(Inventory.station) \
                .order_by(Inventory.station)
            station_count = session \
                .query(Inventory.station) \
                .count()
            print('Station name:')
            print([stat[0] for stat in station], '\n')
            print(f'Total {station_count} stations\n')

            boundary = session \
                .query(sqlalchemy.func.min(Inventory.longitude),
                       sqlalchemy.func.max(Inventory.longitude),
                       sqlalchemy.func.min(Inventory.latitude),
                       sqlalchemy.func.max(Inventory.latitude)) \
                .all()
            print('Station boundary:')
            print(f'West: {boundary[0][0]:>8.4f}')
            print(f'East: {boundary[0][1]:>8.4f}')
            print(f'South: {boundary[0][2]:>7.4f}')
            print(f'North: {boundary[0][3]:>7.4f}\n')

    def plot_map(self):
        """
        Plots station and event map.
        """

        with self.session_scope() as session:
            geometry = session \
                .query(Inventory.latitude,
                       Inventory.longitude,
                       Inventory.network) \
                .all()
            events = session \
                .query(Event.latitude,
                       Event.longitude) \
                .all()

        plot.plot_map(geometry, events)

    def add_events(self, catalog, tag, remove_duplicates=True):
        """
        Add event data form catalog.

        :param str catalog: Catalog name.
        :param str tag: Pick tag.
        :param bool remove_duplicates: Removes duplicates in event table.
        """

        events = io.read_event_list(catalog)
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
            if from_time:
                query = query.filter(Pick.time >= from_time)
            if to_time:
                query = query.filter(Pick.time <= to_time)
            if station:
                station = self.replace_sql_wildcard(station)
                query = query.filter(Pick.station.like(station))
            if phase:
                query = query.filter(Pick.phase.like(phase))
            if tag:
                query = query.filter(Pick.tag.like(tag))

        return query

    def event_summery(self):
        """
        Prints summery from event table.
        """
        with self.session_scope() as session:
            time = session \
                .query(sqlalchemy.func.min(Event.time),
                       sqlalchemy.func.max(Event.time)) \
                .all()
            print('Event time duration:')
            print(f'From: {time[0][0].isoformat()}')
            print(f'To:   {time[0][1].isoformat()}\n')

            event_count = session.query(Event).count()
            print(f'Total {event_count} events\n')

            boundary = session \
                .query(sqlalchemy.func.min(Event.longitude),
                       sqlalchemy.func.max(Event.longitude),
                       sqlalchemy.func.min(Event.latitude),
                       sqlalchemy.func.max(Event.latitude)) \
                .all()
            print('Event boundary:')
            print(f'West: {boundary[0][0]:>8.4f}')
            print(f'East: {boundary[0][1]:>8.4f}')
            print(f'South: {boundary[0][2]:>7.4f}')
            print(f'North: {boundary[0][3]:>7.4f}\n')

    def pick_summery(self):
        """
        Prints summery from pick table.
        """
        with self.session_scope() as session:
            time = session \
                .query(sqlalchemy.func.min(Pick.time),
                       sqlalchemy.func.max(Pick.time)) \
                .all()
            print('Pick time duration:')
            print(f'From: {time[0][0].isoformat()}')
            print(f'To:   {time[0][1].isoformat()}\n')

            print('Phase count:')
            phase_group_count = session \
                .query(Pick.phase, sqlalchemy.func.count(Pick.phase)) \
                .group_by(Pick.phase) \
                .all()
            ps_picks = 0
            for phase, count in phase_group_count:
                if phase in ['P', 'S']:
                    ps_picks += count
                print(f'{count} "{phase}" picks')
            print(f'Total {ps_picks} P + S picks\n')

            station_count = session \
                .query(Pick.station.distinct()) \
                .count()
            print(f'Picks cover {station_count} stations:')

            station = session \
                .query(Pick.station.distinct()) \
                .order_by(Pick.station) \
                .all()
            print([stat[0] for stat in station], '\n')

            no_pick_station = session \
                .query(Inventory.station) \
                .order_by(Inventory.station) \
                .filter(Inventory.station
                        .notin_(session.query(Pick.station))) \
                .all()
            if no_pick_station:
                print(f'{len(no_pick_station)} stations without picks:')
                print([stat[0] for stat in no_pick_station], '\n')

            no_geom_station = session \
                .query(Pick.station.distinct()) \
                .order_by(Pick.station) \
                .filter(Pick.station
                        .notin_(session.query(Inventory.station))) \
                .all()
            if no_geom_station:
                print(f'{len(no_geom_station)} stations without geometry:')
                print([stat[0] for stat in no_geom_station], '\n')

    def generate_training_data(self, pick_list, dataset, chunk_size):
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
        ds = io.read_dataset(dataset)
        index = 0
        with self.session_scope() as session:
            for example in ds:
                instance = core.Instance(example)
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
            if from_time:
                query = query.filter(from_time <= Waveform.endtime)
            if to_time:
                query = query.filter(Waveform.starttime <= to_time)
            if station:
                station = self.replace_sql_wildcard(station)
                query = query.filter(Waveform.station.like(station))

        return query

    def waveform_summery(self):
        """
        Prints summery from waveform table.
        """
        with self.session_scope() as session:
            time = session \
                .query(sqlalchemy.func.min(Waveform.starttime),
                       sqlalchemy.func.max(Waveform.endtime)) \
                .all()
            print('Event time duration:')
            print(f'From: {time[0][0].isoformat()}')
            print(f'To:   {time[0][1].isoformat()}\n')

            event_count = session.query(Waveform).count()
            print(f'Total {event_count} events\n')

            station_count = session \
                .query(Waveform.station.distinct()) \
                .count()
            print(f'Picks cover {station_count} stations:')

            station = session \
                .query(Waveform.station.distinct()) \
                .order_by(Waveform.station) \
                .all()
            print([stat[0] for stat in station], '\n')

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

    def list_distinct_items(self, table, column):
        """
        Returns a query of unique items.

        :param str table: Target table name.
        :param str column: Target column name.
        :rtype: list
        :return: A list of query.
        """
        table = self.get_table_class(table)
        with self.session_scope() as session:
            col = operator.attrgetter(column)
            query = session \
                .query(col(table).distinct()) \
                .order_by(col(table)) \
                .all()
            query = [q[0] for q in query]
            return query

    def clear_table(self, table):
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


if __name__ == "__main__":
    pass
