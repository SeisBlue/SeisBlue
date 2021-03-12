import itertools
import os

import numpy as np
import obspy
from obspy import UTCDateTime

import seisnn.core
import seisnn.example_proto
import seisnn.io
import seisnn.utils


class ExampleGen:
    """
    Main class for Example Generator.

    Consumes data from external source and emit TFRecord.
    """

    def __init__(self,
                 phase=('P', 'S', 'N'),
                 trace_length=30,
                 shape='triang'):
        self.phase = phase
        self.trace_length = trace_length
        self.shape = shape

    def generate_training_data(self,
                               pick_list,
                               tag,
                               database):
        """
        Generate TFRecords from database.

        :param pick_list: List of picks from Pick SQL query.
        :param str tag: Pick tag in SQL database.
        :param str database: SQL database name.
        """
        config = seisnn.utils.Config()
        pick_list = sorted(pick_list,
                           key=lambda pick: [pick.station, pick.time])
        pick_groupby = itertools.groupby(
            pick_list,
            key=lambda pick: [pick.station, UTCDateTime(pick.time).julday])
        group_picks = [[item for item in data] for (key, data) in pick_groupby]

        for index, picks in enumerate(group_picks):
            instance_list = seisnn.utils.parallel(picks,
                                                  func=self.get_instance_list,
                                                  tag=tag,
                                                  database=database)
            flatten = itertools.chain.from_iterable
            flat_list = list(flatten(flatten(instance_list)))
            feature_list = [instance.to_feature() for instance in flat_list]
            example_list = [seisnn.example_proto.feature_to_example(feature)
                            for feature in feature_list]

            net, sta, loc, chan = flat_list[0].metadata.id.split('.')
            year = str(flat_list[0].metadata.starttime.year)
            julday = str(flat_list[0].metadata.starttime.julday)

            tfr_dir = os.path.join(config.tfrecord, year, net, sta)
            seisnn.utils.make_dirs(tfr_dir)
            file_name = f'{net}.{sta}.{loc}.{chan[0:2]}.{year}.{julday}.tfrecord'

            save_file = os.path.join(tfr_dir, file_name)
            seisnn.io.write_tfrecord(example_list, save_file)
            print(f'output {file_name} ({index + 1}/{len(group_picks)})')

    def get_instance_list(self, pick, tag, database):
        """
        Returns instance list form list of picks and SQL database.

        :param pick: List of picks.
        :param str tag: Pick tag in SQL database.
        :param str database: SQL database root.
        :return:
        """

        metadata = self.get_time_window(anchor_time=pick.time,
                                        station=pick.station,
                                        shift='random')

        streams = seisnn.io.read_sds(metadata)
        instance_list = []
        for _, stream in streams.items():
            stream = self.signal_preprocessing(stream)
            instance = seisnn.core.Instance(stream)

            instance.label = seisnn.core.Label(instance.metadata, self.phase)
            instance.label.generate_label(database, tag, self.shape)

            instance.predict = seisnn.core.Label(instance.metadata, self.phase)

            instance_list.append(instance)
        return instance_list

    def get_time_window(self, anchor_time, station, shift=0):
        """
        Returns metadata from anchor time.

        :param anchor_time: Anchor of the time window.
        :param str station: Station name.
        :param float or str shift: (Optional.) Shift in sec,
            if 'random' will shift randomly within the trace length.
        :rtype: dict
        :return: Metadata object.
        """
        if shift == 'random':
            rng = np.random.default_rng()
            shift = rng.random() * self.trace_length

        metadata = seisnn.core.Metadata()
        metadata.starttime = obspy.UTCDateTime(anchor_time) - shift
        metadata.endtime = metadata.starttime + self.trace_length
        metadata.station = station

        return metadata

    def signal_preprocessing(self, stream):
        """
        Return a signal processed stream.

        :param obspy.Stream stream: Stream object.
        :rtype: obspy.Stream
        :return: Processed stream.
        """
        stream.detrend('demean')
        stream.detrend('linear')
        stream.normalize()
        stream.resample(100)
        stream = self.trim_trace(stream)
        return stream

    @staticmethod
    def trim_trace(stream, points=3008):
        """
        Return trimmed stream in a given length.

        :param obspy.Stream stream: Stream object.
        :param int points: Trace data length.
        :rtype: obspy.Stream
        :return: Trimmed stream.
        """

        trace = stream[0]
        start_time = trace.stats.starttime
        dt = (trace.stats.endtime - trace.stats.starttime) / (
                trace.data.size - 1)
        end_time = start_time + dt * (points - 1)
        stream.trim(start_time,
                    end_time,
                    nearest_sample=True,
                    pad=True,
                    fill_value=0)
        return stream
