# -*- coding: utf-8 -*-
import seisblue

import operator
import datetime as dt
import sqlalchemy
import sqlalchemy.ext.declarative
import contextlib
from tqdm import tqdm
from pymysql.err import OperationalError
import pymysql
import time
from operator import itemgetter
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Client:
    """
    Client for sql database
    """

    def __init__(self, database, echo=False, build=False):
        self.database = database
        db_path = database
        username = "****"
        password = "******"
        host = "***.***.***.***"
        port = "****"

        conn = pymysql.connect(
            host=host,
            user=username,
            password=password
        )
        cursor = conn.cursor()

        if build:
            self.engine = sqlalchemy.create_engine(
                f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")
            cursor.execute(f"CREATE DATABASE {database}")  # create db
            cursor.execute(f"USE {database}")
            seisblue.core.Base.metadata.create_all(bind=self.engine)
        try:
            self.engine = sqlalchemy.create_engine(
                f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}",
                echo=echo,
                pool_size=300,
                max_overflow=-1,
            )
            self.session = sqlalchemy.orm.sessionmaker(bind=self.engine)
        except Exception as err:
            print(err)
            raise FileNotFoundError(f"{db_path} is not found")

    def __repr__(self):
        return f"SQL Database({self.database})"

    def show_table_names(self):
        insp = sqlalchemy.inspect(self.engine)
        print(f"Table names : {insp.get_table_names()}")

    def add_inventory(self, objs, remove_duplicates=True):
        with self.session_scope() as session:
            seisblue.core.InventorySQL.__table__.create(bind=session.get_bind(), checkfirst=True)
            try:
                session.bulk_save_objects(objs)
                print(f'Add {len(objs)} inventory into database.')
            except OperationalError as e:
                print(e)
                session.rollback()
        if remove_duplicates:
            self.remove_duplicates('inventory',
                                   ['station', 'latitude', 'longitude', 'elevation', 'starttime', 'endtime'])

    def add_events(self, objs, remove_duplicates=True):
        with self.session_scope() as session:
            seisblue.core.EventSQL.__table__.create(bind=session.get_bind(), checkfirst=True)
            try:
                session.bulk_save_objects(objs)
                print(f'Add {len(objs)} events into database.')
            except OperationalError as e:
                print(e)
                session.rollback()
        if remove_duplicates:
            self.remove_duplicates('event', ['time', 'latitude', 'longitude', 'depth'])

    def add_picks(self, objs, remove_duplicates=True):
        with self.session_scope() as session:
            seisblue.core.PickSQL.__table__.create(bind=session.get_bind(),
                                          checkfirst=True)
            try:
                session.bulk_save_objects(objs)
                print(f'Add {len(objs)} picks into database.')
            except OperationalError as e:
                print(e)
                session.rollback()
        if remove_duplicates:
            self.remove_duplicates('pick', ['time', 'network', 'station', 'phase', 'tag'])

    def add_waveforms(self, objs, remove_duplicates=True):
        with self.session_scope() as session:
            seisblue.core.WaveformSQL.__table__.create(bind=session.get_bind(), checkfirst=True)
            try:
                session.bulk_save_objects(objs)
                print(f'Add {len(objs)} waveforms into database.')
            except OperationalError as e:
                print(e)
                session.rollback()
        if remove_duplicates:
            self.remove_duplicates('waveform', ['starttime', 'endtime', 'network', 'station', 'channel'])

    # def add_pickassoc(self, objs):
    #     with self.session_scope() as session:
    #         seisblue.core.PicksAssoc.__table__.create(bind=session.get_bind(), checkfirst=True)
    #         session.bulk_save_objects(objs)
    #         session.commit()
    #         index_sql = sqlalchemy.DDL('CREATE INDEX idx_time ON pick_assoc (time)')
    #         session.execute(index_sql)
    #         session.commit()
    #
    #     print(f'Add {len(objs)} pickassoc into database.')

    def add_magnitude_into_event(self, event):
        with self.session_scope() as session:
            session.query(event).get(event.id).update({'magnitude': event.magnitude}, synchronize_session=False)

    @staticmethod
    def get_table_class(table):
        """
        Returns related class from dict.

        :param str table: Keywords: inventory, event, pick, waveform.
        :return: SQL table class.
        """
        table_dict = {
            "inventory": seisblue.core.InventorySQL,
            "event": seisblue.core.EventSQL,
            "pick": seisblue.core.PickSQL,
            "waveform": seisblue.core.WaveformSQL,
        }
        try:
            table_class = table_dict.get(table)
            return table_class

        except KeyError:
            msg = "Please select table: inventory, event, pick, waveform"
            raise KeyError(msg)

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
                .query(sqlalchemy.func.min(table.id)) \
                .group_by(*table_columns) \
                .subquery()
            duplicate = session \
                .query(table) \
                .filter(table.id.notin_(session.query(distinct))) \
                .delete(synchronize_session='fetch')

            if duplicate:
                print(f'Remove {duplicate} duplicate {table.__tablename__}s')

    def get_inventory(self, **kwargs):
        """
        Returns query from inventory table.
        :param session
        :rtype: sqlalchemy.orm.query.Query
        :return: A Query.
        """
        inventory = seisblue.core.InventorySQL
        with self.session_scope() as session:
            query = session.query(inventory)
            for key, value in kwargs.items():
                if key == 'network':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(inventory.network.in_(value))
                if key == 'station':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(inventory.station.in_(value))
                if key == 'time':
                    value = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
                    query = query.filter(
                        sqlalchemy.and_(
                            value >= inventory.starttime,
                            value <= inventory.endtime
                        )
                    )
        return query.first()

    def get_picks(self, with_inventory=False, **kwargs):
        pick = seisblue.core.PickSQL
        inventory = seisblue.core.InventorySQL
        with self.session_scope() as session:
            if with_inventory:
                query = session.query(pick, inventory).join(inventory, pick.station == inventory.station)
            else:
                query = session.query(pick)
            for key, value in kwargs.items():
                if key == 'from_time':
                    value = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
                    query = query.filter(pick.time >= value)
                elif key == 'to_time':
                    value = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
                    query = query.filter(pick.time <= value)
                elif key == 'network':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(pick.network.in_(value))
                elif key == 'station':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(pick.station.in_(value))
                elif key == 'phase':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(pick.phase.in_(value))
                elif key == 'tag':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(pick.tag.in_(value))
                elif key == 'low_snr':
                    query = query.filter(pick.snr >= value)
                elif key == 'high_snr':
                    query = query.filter(pick.snr <= value)
                elif key == 'low_confidence':
                    query = query.filter(pick.confidence >= value)
                elif key == 'high_confidence':
                    query = query.filter(pick.confidence <= value)
        return query.all()


    def get_events(
            self,
            from_time=None,
            to_time=None,
            min_longitude=None,
            max_longitude=None,
            min_latitude=None,
            max_latitude=None,
            min_depth=None,
            max_depth=None,
    ):
        with self.session_scope() as session:
            query = session.query(seisblue.core.EventSQL)
            if from_time is not None:
                query = query.filter(seisblue.core.EventSQL.time >= from_time)
            if to_time is not None:
                query = query.filter(seisblue.core.EventSQL.time <= to_time)
            if min_longitude is not None:
                query = query.filter(seisblue.core.EventSQL.longitude >= min_longitude)
            if max_longitude is not None:
                query = query.filter(seisblue.core.EventSQL.longitude <= max_longitude)
            if min_latitude is not None:
                query = query.filter(seisblue.core.EventSQL.latitude >= min_latitude)
            if max_latitude is not None:
                query = query.filter(seisblue.core.EventSQL.latitude <= max_latitude)
            if min_depth is not None:
                query = query.filter(seisblue.core.EventSQL.depth >= min_depth)
            if max_depth is not None:
                query = query.filter(seisblue.core.EventSQL.depth <= max_depth)
        return query.all()

    def get_waveform(self, **kwargs):
        waveform = seisblue.core.WaveformSQL
        with self.session_scope() as session:
            query = session.query(seisblue.core.WaveformSQL)
            for key, value in kwargs.items():
                if key == 'time':
                    query = query.filter(
                        sqlalchemy.and_(
                            value >= seisblue.core.WaveformSQL.starttime,
                            value <= seisblue.core.WaveformSQL.endtime
                        )
                    )
                elif key == 'from_time':
                    value = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
                    query = query.filter(waveform.starttime >= value)
                elif key == 'to_time':
                    value = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
                    query = query.filter(waveform.starttime <= value)
                elif key == 'station':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(seisblue.core.WaveformSQL.station.in_(value))
                elif key == 'network':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(seisblue.core.WaveformSQL.network.in_(value))
                elif key == 'dataset':
                    value = [value] if isinstance(value, str) else value
                    query = query.filter(seisblue.core.WaveformSQL.dataset.in_(value))
        return query.all()

    def get_pick_assoc_id(self, assoc_id):
        with self.session_scope() as session:
            query = session.query(seisblue.core.PickAssoc).filter(seisblue.core.PickAssoc.assoc_id == assoc_id)
        return query.all()

    def get_candidate(self, assoc_id, commit=False):
        origin_time_delta = 3
        origin_time_delta = dt.timedelta(seconds=origin_time_delta)
        nsta_declare = 3
        with self.session_scope() as session:
            candidates = (
                session.query(seisblue.core.Candidate)
                .filter(seisblue.core.Candidate.assoc_id == assoc_id)
                .all()
            )
            print(candidates)

            event = self.assoc_parallel(
                peak_candidates=candidates,
                origin_time_delta=origin_time_delta,
                nsta_declare=nsta_declare,
            )
            print(event)
            for candidate in candidates:
                inventory = (
                    session.query(seisblue.core.InventorySQL)
                    .filter(seisblue.core.InventorySQL.station == candidate.sta)
                    .all()
                )
                distance = seisblue.utils.calculate_distance(inventory[0].longitude,
                                                    inventory[0].latitude,
                                                    event.longitude,
                                                    event.latitude)
                candidate.distance = distance
                # session.commit()


    def assoc_parallel(self, peak_candidates, origin_time_delta, nsta_declare):
        with self.session_scope() as session:
            nsta = len(peak_candidates)
            print(
                f"time: {str(peak_candidates[0].origin_time)}, station count: {nsta}")

            new_match_candidates, origin, QA = seisblue.assoc_seisblue.utils.hypo_search(
                peak_candidates)
            print(f"Hypocenter searching done, result: {QA}")

            if not QA:
                return

            if len(new_match_candidates) < nsta_declare:
                print("no enough picks left by hypo search")
                return

            nsta = len(new_match_candidates)
            new_event = seisblue.core.AssociatedEvent(
                origin.time.datetime,
                0,
                origin.latitude,
                origin.longitude,
                origin.depth,
                nsta,
                origin.time_errors.uncertainty,
                origin.longitude_errors.uncertainty,
                origin.latitude_errors.uncertainty,
                origin.depth_errors.uncertainty,
                origin.quality.azimuthal_gap,
            )
        return new_event

    def analysis_candidate(self, assoc_id):
        with self.session_scope() as session:
            candidates = (
                session.query(seisblue.core.Candidate)
                .filter(seisblue.core.Candidate.assoc_id == assoc_id)
                .all()
            )
            data = [item.__dict__ for item in candidates]
            df = pd.DataFrame(data).drop('_sa_instance_state', axis=1)

            for column in ['origin_time', 'p_time', 's_time']:
                df[column] = pd.to_datetime(df[column])
            selected_columns = ['origin_time', 'p_time', 's_time', 'distance']
            df_selected = df[selected_columns]

            start_time = df_selected['origin_time'].min()
            for column in ['origin_time', 'p_time', 's_time']:
                df_selected[column] = (
                            df_selected[column] - start_time).dt.total_seconds()

            df_selected['p-s'] = df_selected['s_time'] - df_selected['p_time']

            sns.pairplot(df_selected)

            figdir = './figure/check_attr_relation'
            seisblue.utils.check_dir(figdir)
            plt.savefig(f'{figdir}/{assoc_id}.jpg')

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
            query = session.query(col(table).distinct()).order_by(
                col(table)).all()
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
            query = (
                session.query(col(table))
                .order_by(col(table))
                .filter(col(table).notin_(exclude))
                .all()
            )
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
            print(f"{exception.__class__.__name__}: {exception.__cause__}")
            session.rollback()
        finally:
            session.close()


class DatabaseInspector:
    """
    Main class for Database Inspector.
    """

    def __init__(self, database):
        if isinstance(database, str):
            try:
                database = Client(database)
            except Exception as exception:
                print(f"{exception.__class__.__name__}: {exception.__cause__}")

        self.database = database

    def inventory_summery(self):
        """
        Prints summery from geometry table.
        """
        stations = self.database.get_distinct_items("inventory", "station")

        print("Station name:")
        print([station for station in stations], "\n")
        print(f"Total {len(stations)} stations\n")

        latitudes = self.database.get_distinct_items("inventory", "latitude")
        longitudes = self.database.get_distinct_items("inventory", "longitude")

        print("Station boundary:")
        print(f"West: {min(longitudes):>8.4f}")
        print(f"East: {max(longitudes):>8.4f}")
        print(f"South: {min(latitudes):>7.4f}")
        print(f"North: {max(latitudes):>7.4f}\n")

    def event_summery(self):
        """
        Prints summery from event table.
        """
        times = self.database.get_distinct_items("event", "time")
        print("Event time duration:")
        print(f"From: {min(times).isoformat()}")
        print(f"To:   {max(times).isoformat()}\n")

        events = self.database.get_events()
        print(f"Total {len(events)} events\n")

        latitudes = self.database.get_distinct_items("event", "latitude")
        longitudes = self.database.get_distinct_items("event", "longitude")

        print("Station boundary:")
        print(f"West: {min(longitudes):>8.4f}")
        print(f"East: {max(longitudes):>8.4f}")
        print(f"South: {min(latitudes):>7.4f}")
        print(f"North: {max(latitudes):>7.4f}\n")

    def pick_summery(self):
        """
        Prints summery from pick table.
        """
        times = self.database.get_distinct_items("pick", "time")
        print("Event time duration:")
        print(f"From: {min(times).isoformat()}")
        print(f"To:   {max(times).isoformat()}\n")

        print("Phase count:")
        phases = self.database.get_distinct_items("pick", "phase")
        for phase in phases:
            picks = self.database.get_picks(phase=phase)
            print(f'{len(picks)} "{phase}" picks')
        print()

        pick_stations = self.database.get_distinct_items("pick", "station")
        print(f"Picks cover {len(pick_stations)} stations:")
        print([station for station in pick_stations], "\n")

        no_pick_station = self.database.get_exclude_items(
            "inventory", "station", pick_stations
        )
        if no_pick_station:
            print(f"{len(no_pick_station)} stations without picks:")
            print([station for station in no_pick_station], "\n")

        inventory_station = self.database.get_distinct_items("inventory", "station")
        no_inventory_station = self.database.get_exclude_items(
            "pick", "station", inventory_station
        )

        if no_inventory_station:
            print(f"{len(no_inventory_station)} stations without geometry:")
            print([station for station in no_inventory_station], "\n")

    def waveform_summery(self):
        """
        Prints summery from waveform table.
        """
        starttimes = self.database.get_distinct_items("waveform", "starttime")
        endtimes = self.database.get_distinct_items("waveform", "endtime")
        print("Waveform time duration:")
        print(f"From: {min(starttimes).isoformat()}")
        print(f"To:   {max(endtimes).isoformat()}\n")

        waveforms = self.database.get_events()
        print(f"Total {len(waveforms)} events\n")

        stations = self.database.get_distinct_items("waveform", "station")
        print(f"Picks cover {len(stations)} stations:")
        print([station for station in stations], "\n")

    def plot_map(self, **kwargs):
        """
        Plots station and event map.
        """
        inventories = self.database.get_inventories()
        events = self.database.get_events()
        seisblue.plot.plot_map(inventories, events, **kwargs)

if __name__ == '__main__':
    pass
