"""
Core
"""
import os

import numpy as np
import scipy.signal
import obspy

import seisblue.example_proto
import seisblue.io
import seisblue.plot
import seisblue.sql


class Metadata:
    """
    Main class for metadata.
    """
    id = None
    station = None

    starttime = None
    endtime = None
    npts = None
    delta = None

    def __init__(self, input_data=None):
        if isinstance(input_data, obspy.Trace):
            self.from_trace(input_data)

        elif isinstance(input_data, seisblue.example_proto.Feature):
            self.from_feature(input_data)

    def from_trace(self, trace):
        self.id = trace.id
        self.station = trace.stats.station

        self.starttime = trace.stats.starttime
        self.endtime = trace.stats.endtime
        self.npts = trace.stats.npts
        self.delta = trace.stats.delta
        return self

    def from_feature(self, feature):
        self.id = feature.id
        self.station = feature.station

        self.starttime = obspy.UTCDateTime(feature.starttime)
        self.endtime = obspy.UTCDateTime(feature.endtime)
        self.npts = feature.npts
        self.delta = feature.delta
        return self


class Trace:
    """
    Main class for trace data.
    """
    metadata = None
    channel = None
    data = None

    def __init__(self, input_data):
        if isinstance(input_data, obspy.Stream):
            self.from_stream(input_data)

        elif isinstance(input_data, seisblue.example_proto.Feature):
            self.from_feature(input_data)

    def from_stream(self, stream):
        """
        Gets waveform from Obspy stream.

        :param stream: Obspy stream object.
        :return: Waveform object.
        """
        channel = []
        data = np.zeros([stream[0].stats.npts, 3])
        for i, comp in enumerate(['E', 'N', 'Z']):
            try:
                st = stream.select(component=comp)
                data[:, i] = st.traces[0].data
                channel.append(st.traces[0].stats.channel)

            except IndexError:
                pass

            except Exception as error:
                print(f'{type(error).__name__}: {error}')

        self.data = data
        self.channel = channel
        self.metadata = Metadata(stream.traces[0])

        return self

    def from_feature(self, feature):
        self.metadata = Metadata(feature)
        self.data = feature.trace
        self.channel = feature.channel
        return self

    def get_snr(self, pick, second=1):
        vector = np.linalg.norm(self.data, axis=2)[0]
        point = int((pick.time - self.metadata.starttime) * 100)
        if point >= second * 100:
            signal = vector[point:point + second * 100]
            noise = vector[point - len(signal):point]
        else:
            noise = vector[0:point]
            signal = vector[point:point + len(noise)]
        snr = seisblue.qc.signal_to_noise_ratio(signal=signal, noise=noise)
        pick.snr = np.around(snr, 4)


class Label:
    """
    Main class for label data.
    """
    picks = None

    def __init__(self, metadata, phase, tag=None):
        self.metadata = metadata
        self.phase = phase
        self.tag = tag
        self.data = np.zeros([metadata.npts, len(phase)])

    def generate_label(self, database, tag, shape, half_width=20):
        """
        Add generated label to stream.

        :param str database: SQL database.
        :param str tag: Pick tag in SQL database.
        :param str shape: Label shape, see scipy.signal.windows.get_window().
        :param int half_width: Label half width in data point.
        :rtype: np.array
        :return: Label.
        """
        db = seisblue.sql.Client(database)

        ph_index = {}
        for i, phase in enumerate(self.phase):
            ph_index[phase] = i
            picks = db.get_picks(from_time=self.metadata.starttime.datetime,
                                 to_time=self.metadata.endtime.datetime,
                                 station=self.metadata.station,
                                 phase=phase, tag=tag)

            for pick in picks:
                pick_time = obspy.UTCDateTime(
                    pick.time) - self.metadata.starttime
                pick_time_index = int(pick_time / self.metadata.delta)
                self.data[pick_time_index, i] = 1

        if 'EQ' in self.phase:
            self.data[:, ph_index['EQ']] = \
                self.data[:, ph_index['P']] - self.data[:, ph_index['S']]

            self.data[:, ph_index['EQ']] = \
                np.cumsum(self.data[:, ph_index['EQ']])

            if np.any(self.data[:, ph_index['EQ']] < 0):
                self.data[:, ph_index['EQ']] += 1

        for i, phase in enumerate(self.phase):
            if not phase == 'EQ':
                wavelet = scipy.signal.windows.get_window(
                    shape, 2 * half_width)
                self.data[:, i] = scipy.signal.convolve(
                    self.data[:, i], wavelet[1:], mode='same')

        if 'N' in self.phase:
            # Make Noise window by 1 - P - S
            self.data[:, ph_index['N']] = 1
            self.data[:, ph_index['N']] -= self.data[:, ph_index['P']]
            self.data[:, ph_index['N']] -= self.data[:, ph_index['S']]

        return self

    def get_picks(self, threshold=0.5, distance=100, from_ms=0, to_ms=-1):
        """
        Extract pick from label and write into the database.

        :param float height: Height threshold, from 0 to 1, default is 0.5.
        :param int distance: Distance threshold in data point.
        """
        picks = []
        for i, phase in enumerate(self.phase[0:2]):
            peaks, properties = scipy.signal.find_peaks(
                self.data[-1, from_ms:to_ms, i],
                height=threshold,
                distance=distance)

            for j, peak in enumerate(peaks):
                if peak:
                    pick_time = obspy.UTCDateTime(self.metadata.starttime) \
                                + (peak + from_ms) * self.metadata.delta

                    picks.append(Pick(time=pick_time,
                                      station=self.metadata.station,
                                      phase=self.phase[i],
                                      confidence=round(
                                          float(properties['peak_heights'][j]), 2))
                                 )

        self.picks = picks

    def write_picks_to_database(self, tag,trace_id, database):
        """
        Write picks into the database.

        :param str tag: Output pick tag name.
        :param database: SQL database name.
        """
        db = seisblue.sql.Client(database)
        for pick in self.picks:
            db.add_pick(time=pick.time.datetime,
                        station=pick.station,
                        phase=pick.phase,
                        tag=tag,
                        snr=pick.snr,
                        confidence=pick.confidence,
                        trace_id = trace_id)


class Pick:
    """
    Main class for phase pick.
    """

    def __init__(self,
                 time=None,
                 station=None,
                 phase=None,
                 tag=None,
                 snr=None,
                 confidence=None):
        self.time = time
        self.station = station
        self.phase = phase
        self.tag = tag
        self.snr = snr
        self.confidence = confidence


class Instance:
    """
    Main class for data transfer.
    """
    metadata = None
    trace = None
    label = None
    predict = None

    def __init__(self, input_data=None):
        if input_data is None:
            pass

        try:
            if isinstance(input_data, obspy.Stream):
                self.from_stream(input_data)

            elif isinstance(input_data, seisblue.sql.Waveform):
                dataset = seisblue.io.read_dataset(input_data.tfrecord)
                for item in dataset.skip(input_data.data_index).take(1):
                    input_data = item
                self.from_example(input_data)

            else:
                self.from_example(input_data)

        except TypeError:
            pass

        except Exception as error:
            print(f'{type(error).__name__}: {error}')

    def __repr__(self):
        return f"Instance(" \
               f"ID={self.metadata.id}, " \
               f"Start Time={self.metadata.starttime}, " \
               f"Phase={self.label.phase})"

    def from_stream(self, stream):
        """
        Initialized from stream.

        :param stream:
        :return:
        """
        self.trace = Trace(stream)
        self.metadata = self.trace.metadata

        return self

    def from_feature(self, feature):
        """
        Initialized from feature dict.

        :param Feature feature: Feature dict.
        """
        self.trace = Trace(feature)
        self.metadata = self.trace.metadata

        self.label = Label(self.metadata, feature.phase, tag='label')
        self.label.data = feature.label

        self.predict = Label(self.metadata, feature.phase, tag='predict')
        self.predict.data = feature.predict
        return self

    def to_feature(self):
        """
        Returns Feature object.

        :rtype: Feature
        :return: Feature object.
        """
        feature = seisblue.example_proto.Feature()

        feature.id = self.metadata.id
        feature.station = self.metadata.station
        feature.starttime = self.metadata.starttime.isoformat()
        feature.endtime = self.metadata.endtime.isoformat()

        feature.npts = self.metadata.npts
        feature.delta = self.metadata.delta

        feature.trace = self.trace.data
        feature.channel = self.trace.channel

        feature.phase = self.label.phase
        feature.label = self.label.data
        feature.predict = self.predict.data

        return feature

    def from_example(self, example):
        """
        Initialized from example protocol.

        :param example: Example protocol.
        """
        feature = seisblue.example_proto.eval_eager_tensor(example)
        self.from_feature(feature)
        return self

    def to_example(self):
        """
        Returns example protocol.

        :return: Example protocol.
        """
        feature = self.to_feature()
        example = seisblue.example_proto.feature_to_example(feature)
        return example

    def to_tfrecord(self, file_path):
        """
        Write TFRecord to file path.

        :param str file_path: Output path.
        """
        feature = self.to_feature()
        example = seisblue.example_proto.feature_to_example(feature)
        seisblue.io.write_tfrecord([example], file_path)

    def plot(self, **kwargs):
        """
        Plot dataset.

        :param kwargs: Keywords pass into plot.
        """
        seisblue.plot.plot_dataset(self, **kwargs)

    def get_tfrecord_name(self):
        year = str(self.metadata.starttime.year)
        julday = str(self.metadata.starttime.julday)
        return f'{self.metadata.id[:-1]}.{year}.{julday}.tfrecord'

    def get_tfrecord_dir(self, sub_dir):
        """

        :param sub_dir: Sub TFRecord directory: 'train', 'test', 'eval'
        :return: TFRecord directory
        """
        config = seisblue.utils.Config()
        name = self.get_tfrecord_name()
        net, sta, loc, chan, year, julday, suffix = name.split('.')

        sub_dir = getattr(config, sub_dir)
        tfr_dir = os.path.join(sub_dir, year, net, sta)

        return tfr_dir


if __name__ == "__main__":
    pass
