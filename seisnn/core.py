"""
Core
"""
import numpy as np
import scipy.signal
import obspy

import seisnn.example_proto
import seisnn.io
import seisnn.plot
import seisnn.sql


class Instance:
    """
    Main class for data transfer.
    """
    id = None
    station = None

    starttime = None
    endtime = None
    npts = None
    delta = None

    trace = None
    channel = None

    phase = None
    label = None
    predict = None

    def __init__(self, input_data=None):
        if input_data is None:
            pass
        try:
            if isinstance(input_data, seisnn.sql.Waveform):
                dataset = seisnn.io.read_dataset(input_data.dataset)
                for item in dataset.skip(input_data.data_index).take(1):
                    input_data = item

            self.from_example(input_data)
        except TypeError:
            pass

        except Exception as error:
            print(f'{type(error).__name__}: {error}')

    def __repr__(self):
        return f"Instance(" \
               f"ID={self.id}, " \
               f"Start Time={self.starttime}, " \
               f"Phase={self.phase})"

    def from_stream(self, stream):
        """

        :param obspy.Trace stream:
        :return:
        """
        trace = stream.traces[0]

        self.id = trace.id
        self.station = trace.stats.station

        self.starttime = trace.stats.starttime
        self.endtime = trace.stats.endtime
        self.npts = trace.stats.npts
        self.delta = trace.stats.delta

        channel = []
        trace = np.zeros([3008, 3])
        for i, comp in enumerate(['Z', 'N', 'E']):
            try:
                st = stream.select(component=comp)
                trace[:, i] = st.traces[0].data
                channel.append(st.traces[0].stats.channel)
            except IndexError:
                pass

            except Exception as error:
                print(f'{type(error).__name__}: {error}')

        self.trace = trace
        self.channel = channel
        return self

    def from_feature(self, feature):
        """
        Initialized from feature dict.

        :type feature: dict
        :param feature: Feature dict.
        """
        self.id = feature['id']
        self.station = feature['station']

        self.starttime = obspy.UTCDateTime(feature['starttime'])
        self.endtime = obspy.UTCDateTime(feature['endtime'])
        self.npts = feature['npts']
        self.delta = feature['delta']

        self.trace = feature['trace']
        self.channel = feature['channel']

        self.phase = feature['phase']
        self.label = feature['label']
        self.predict = feature['predict']

    def to_feature(self):
        """
        Returns feature dict.

        :rtype: dict
        :return: Feature dict.
        """
        feature = {
            'id': self.id,
            'station': self.station,
            'starttime': self.starttime.isoformat(),
            'endtime': self.endtime.isoformat(),

            'npts': self.npts,
            'delta': self.delta,

            'trace': self.trace,
            'channel': self.channel,

            'phase': self.phase,
            'label': self.label,
            'predict': self.predict,
        }
        return feature

    def from_example(self, example):
        """
        Initialized from example protocol.

        :param example: Example protocol.
        """
        feature = seisnn.example_proto.eval_eager_tensor(example)
        self.from_feature(feature)

    def to_example(self):
        """
        Returns example protocol.

        :return: Example protocol.
        """
        feature = self.to_feature()
        example = seisnn.example_proto.feature_to_example(feature)
        return example

    def to_tfrecord(self, file_path):
        """
        Write TFRecord to file path.

        :param str file_path: Output path.
        """
        feature = self.to_feature()
        example = seisnn.example_proto.feature_to_example(feature)
        seisnn.io.write_tfrecord([example], file_path)

    def get_label(self, database, **kwargs):
        self.label = seisnn.processing.get_label(self, database, **kwargs)

    def plot(self, **kwargs):
        """
        Plot dataset.

        :param kwargs: Keywords pass into plot.
        """
        seisnn.plot.plot_dataset(self, **kwargs)


class TimeFrame:
    """
    Main class for time frame.
    """
    id = None
    station = None

    starttime = None
    endtime = None
    npts = None
    delta = None

    data = None


class Waveform(TimeFrame):
    """
    Main class for waveform.
    """
    channel = None


class CharFunc(TimeFrame):
    """
    Main class for Characteristic function.
    """
    phase = None
    picks = []

    def get_picks(self, height=0.4, distance=100):
        """
        Extract pick from label and write into the database.

        :param float height: Height threshold, from 0 to 1, default is 0.5.
        :param int distance: Distance threshold in data point.
        """
        for i, phase in enumerate(self.phase[0:2]):
            peaks, _ = scipy.signal.find_peaks(
                self.data[-1, :, i],
                height=height,
                distance=distance)

            for peak in peaks:
                if peak:
                    pick_time = obspy.UTCDateTime(self.starttime) \
                                + peak * self.delta

                    self.picks.append(
                        Pick(pick_time, self.station, self.phase[i])
                    )

    def write_picks_to_database(self, tag, database):
        """
        Write picks into the database.

        :param str tag: Output pick tag name.
        :param database: SQL database name.
        """
        db = seisnn.sql.Client(database)
        for pick in self.picks:
            db.add_pick(time=pick.time.datetime,
                        station=pick.station,
                        phase=pick.phase,
                        tag=tag)


class Pick:
    """
    Main class for phase pick.
    """

    def __init__(self, time, station, phase):
        self.time = time
        self.station = station
        self.phase = phase


if __name__ == "__main__":
    pass
