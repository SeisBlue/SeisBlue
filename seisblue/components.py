import itertools
import os

import numpy as np
import obspy
from obspy import UTCDateTime

import seisblue.core
import seisblue.example_proto
import seisblue.io
import seisblue.utils


class TFRecordConverter:
    """
    Main class for TFRecord Converter.

    Consumes data from external source and emit TFRecord.
    """

    def __init__(self,
                 phase=('P', 'S', 'N'),
                 trace_length=30,
                 shape='triang'):
        self.phase = phase
        self.trace_length = trace_length
        self.shape = shape

    def convert_training_from_picks(self, pick_list, tag, database,
                                    cpu_count=None):
        """
        Convert training TFRecords from database picks.

        :param pick_list: List of picks from Pick SQL query.
        :param str tag: Pick tag in SQL database.
        :param str database: SQL database name.
        """
        pick_list = sorted(pick_list,
                           key=lambda pick: [pick.station, pick.time])
        pick_groupby = itertools.groupby(
            pick_list,
            key=lambda pick: [pick.station, UTCDateTime(pick.time).julday])
        group_picks = [[item for item in data] for (key, data) in pick_groupby]

        seisblue.utils.parallel(group_picks,
                                func=self.write_tfrecord,
                                sub_dir='train',
                                tag=tag,
                                database=database,
                                batch_size=1,
                                cpu_count=cpu_count)

    def write_tfrecord(self, picks, sub_dir, tag, database):
        instance_list = self.get_instance_list(picks, tag, database)
        if instance_list:
            feature_list = [instance.to_feature() for instance in
                            instance_list]
            example_list = [seisblue.example_proto.feature_to_example(feature)
                            for feature in feature_list]

            tfr_dir = instance_list[0].get_tfrecord_dir(sub_dir)
            seisblue.utils.make_dirs(tfr_dir)

            file_name = instance_list[0].get_tfrecord_name()
            save_file = os.path.join(tfr_dir, file_name)

            seisblue.io.write_tfrecord(example_list, save_file)

            print(f'output {file_name}')

    def get_instance_list(self, picks, tag, database):
        """
        Returns instance list form list of picks and SQL database.

        :param picks: List of picks.
        :param str tag: Pick tag in SQL database.
        :param str database: SQL database root.
        :return:
        """
        metadata = self.get_time_window(anchor_time=UTCDateTime(0),
                                        station='',
                                        shift='random')
        instance_list = []
        try:
            for pick in picks:
                if metadata.starttime < pick.time < metadata.endtime:
                    continue
                elif metadata.endtime < pick.time < metadata.endtime + 30:
                    metadata = self.get_time_window(
                        anchor_time=metadata.starttime + 30,
                        station=pick.station,
                    )
                else:
                    metadata = self.get_time_window(anchor_time=pick.time,
                                                    station=pick.station,
                                                    shift='random')

                streams = seisblue.io.read_sds(metadata)

                for _, stream in streams.items():
                    stream = self.signal_preprocessing(stream)
                    instance = seisblue.core.Instance(stream)

                    instance.label = seisblue.core.Label(instance.metadata,
                                                         self.phase)
                    instance.label.generate_label(database, tag, self.shape)

                    instance.predict = seisblue.core.Label(instance.metadata,
                                                           self.phase)

                    instance_list.append(instance)
        except Exception as e:
            print(f'station = {pick.station}, time = {pick.time}, error = {e}')
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
            shift = rng.random() * (self.trace_length - 3)
            shift = shift + 3
        metadata = seisblue.core.Metadata()
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
        stream.filter('bandpass', freqmin=1, freqmax=45)
        for i in stream:
            i.normalize()
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
        if trace.data.size > 1:
            dt = (trace.stats.endtime - trace.stats.starttime) / (
                    trace.data.size - 1)
            end_time = start_time + 3 + dt * (points - 1)
        elif trace.data.size == 1:
            end_time = start_time
        else:
            print('No data points in trace')
            return
        stream.trim(start_time + 3,
                    end_time,
                    nearest_sample=True,
                    pad=True,
                    fill_value=0)
        return stream
