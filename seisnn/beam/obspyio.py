import os

import apache_beam as beam
import numpy as np
import scipy
import scipy.stats as ss
import tensorflow as tf
from obspy import Stream
from obspy.clients.filesystem.sds import Client
from obspy.core.inventory.util import Latitude, Longitude
from obspy.io.nordic.core import read_nordic


class StreamFeatureExtraction(beam.DoFn):
    def process(self, stream, *args, **kwargs):
        trace = stream[0]

        feature = {
            'starttime': trace.stats.starttime.isoformat(),
            'endtime': trace.stats.endtime.isoformat(),
            'station': trace.stats.station,
            'npts': trace.stats.npts,
            'delta': trace.stats.delta,

            'latitude': stream.location['latitude'],
            'longitude': stream.location['longitude'],
            'elevation': stream.location['elevation'],
        }

        channel_list = []
        for trace in stream:
            channel = trace.stats.channel
            channel_list.append(channel)
            feature[channel] = trace.data
        feature['channel'] = channel_list

        phase_list = []
        pick_time = []
        pick_phase = []
        pick_type = []

        if stream.picks:
            for phase, picks in stream.picks.items():
                phase_list.append(phase)
                for pick in picks:
                    pick_time.append(pick.time.isoformat())
                    pick_phase.append(pick.phase_hint)
                    pick_type.append(pick.evaluation_mode)

        feature['phase'] = phase_list
        feature['pick_time'] = pick_time
        feature['pick_phase'] = pick_phase
        feature['pick_type'] = pick_type

        for phase, pdf in stream.pdf.items():
            feature[phase] = pdf

        return [feature]


class FeatureToExample(beam.DoFn):
    def process(self, stream, *args, **kwargs):
        context = tf.train.Features(feature={
            'starttime':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[stream['starttime'].encode('utf-8')])),
            'endtime':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[stream['endtime'].encode('utf-8')])),
            'station':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[stream['station'].encode('utf-8')])),
            'npts':  tf.train.Feature(int64_list=tf.train.Int64List(value=[stream['npts']])),
            'delta':  tf.train.Feature(float_list=tf.train.FloatList(value=[stream['delta']])),
            'latitude':  tf.train.Feature(float_list=tf.train.FloatList(value=[stream['latitude']])),
            'longitude':  tf.train.Feature(float_list=tf.train.FloatList(value=[stream['longitude']])),
            'elevation':  tf.train.Feature(float_list=tf.train.FloatList(value=[stream['elevation']]))
        })

        data_dict = {}
        for key in ['channel', 'phase']:
            for sequence_data in stream[key]:
                trace_features = []
                for i in stream[sequence_data]:
                    trace_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[i]))
                    trace_features.append(trace_feature)
                data_dict[sequence_data] = tf.train.FeatureList(feature=trace_features)

        for key in ['pick_time', 'pick_phase', 'pick_type']:
            pick_features = []
            if stream[key]:
                for data in stream[key]:
                    pick_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.encode('utf-8')]))
                    pick_features.append(pick_feature)
            else:
                pick_features.append(tf.train.Feature(bytes_list=tf.train.BytesList(value=['NA'.encode('utf-8')])))

            data_dict[key] = tf.train.FeatureList(feature=pick_features)

        feature_list = tf.train.FeatureLists(feature_list=data_dict)
        example = tf.train.SequenceExample(context=context, feature_lists=feature_list)
        return [example]


class ReadHYP(beam.PTransform):
    def __init__(self, file_patterns):
        super(ReadHYP, self).__init__()
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
        self.file_patterns = file_patterns

    def expand(self, pcollection):
        def get_location(filename):
            location_list = []
            with open(filename, 'r') as file:
                blank_line = 0
                while True:
                    line = file.readline().rstrip()

                    if not len(line):
                        blank_line += 1
                        continue

                    if blank_line > 1:
                        break

                    elif blank_line == 1:
                        lat = line[6:14]
                        lon = line[14:23]
                        elev = float(line[23:])
                        sta = line[1:6].strip()

                        if lat[-1] == 'S':
                            NS = -1
                        else:
                            NS = 1

                        if lon[-1] == 'W':
                            EW = -1
                        else:
                            EW = 1

                        lat = (int(lat[0:2]) + float(lat[2:-1]) / 60) * NS
                        lat = Latitude(lat)

                        lon = (int(lon[0:3]) + float(lon[3:-1]) / 60) * EW
                        lon = Longitude(lon)

                        location = {'station': sta,
                                    'latitude': lat,
                                    'longitude': lon,
                                    'elevation': elev}
                        location_list.append(location)
            return location_list

        return (
                pcollection
                | 'Create filename' >> beam.Create(self.file_patterns)
                | 'Get location' >> beam.FlatMap(get_location)
        )


class ReadSfile(beam.PTransform):
    def __init__(self, file_patterns):
        super(ReadSfile, self).__init__()
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
        self.file_patterns = file_patterns

    def expand(self, pcollection):
        def get_dir_list(file_dir, suffix=""):
            file_list = []
            for file_name in os.listdir(file_dir):
                f = os.path.join(file_dir, file_name)
                if file_name.endswith(suffix):
                    file_list.append(f)

            return file_list

        def get_events(filename):
            catalog, wavename = read_nordic(filename, return_wavnames=True)
            for event in catalog.events:
                for pick in event.picks:
                    pick.waveform_id.wavename = wavename
                yield event

        return (
                pcollection
                | 'Create file directory' >> beam.Create(self.file_patterns)
                | 'List all files' >> beam.FlatMap(get_dir_list)
                | 'Get event' >> beam.FlatMap(get_events)
        )


class GetWindowFromPick(beam.DoFn):
    def __init__(self, trace_length=30):
        super(GetWindowFromPick, self).__init__()
        self.trace_length = trace_length

    def process(self, pick, *args, **kwargs):
        scipy.random.seed()
        pick_time = pick.time

        starttime = pick_time - self.trace_length + np.random.random_sample() * self.trace_length
        endtime = starttime + self.trace_length

        window = {
            'starttime': starttime,
            'endtime': endtime,
            'station': pick.waveform_id.station_code,
            'wavename': pick.waveform_id.wavename
        }
        return [window]


class ReadSDS(beam.DoFn):
    def __init__(self, sds_root):
        super(ReadSDS, self).__init__()
        self.sds_root = sds_root

    def process(self, window, *args, **kwargs):
        station = window['station']
        starttime = window['starttime']
        endtime = window['endtime']

        client = Client(sds_root=self.sds_root)
        stream = client.get_waveforms(network="*", station=station, location="*", channel="*",
                                      starttime=starttime, endtime=endtime)

        stream.sort(keys=['channel'], reverse=True)
        seismometer_list = {}

        for trace in stream:
            current_type = trace.stats.channel[0:2]
            if not seismometer_list.get(current_type):
                seismometer_list[current_type] = Stream(trace)
            else:
                seismometer_list[current_type].append(trace)

        for key, value in seismometer_list.items():
            yield value


class TrimTrace(beam.DoFn):
    def __init__(self, points=3001):
        super(TrimTrace, self).__init__()
        self.points = points

    def process(self, stream, *args, **kwargs):
        trace = stream[0]
        start_time = trace.stats.starttime
        dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
        end_time = start_time + dt * (self.points - 1)
        stream.trim(start_time, end_time, nearest_sample=False, pad=True, fill_value=0)
        return [stream]


class FilterPickPhase(beam.DoFn):
    def __init__(self, phase):
        super(FilterPickPhase, self).__init__()
        self.phase = phase

    def process(self, pick, *args, **kwargs):
        if pick.phase_hint == self.phase:
            return [pick]
        else:
            return


class DropEmptyStation(beam.DoFn):
    def process(self, data, *args, **kwargs):
        station, context = data
        if context['stream']:
            return [data]
        else:
            return


class GroupStreamPick(beam.PTransform):
    def expand(self, pcollection):
        def search_pick(pick_list, stream):
            tmp_pick = {}
            starttime = stream.traces[0].stats.starttime
            endtime = stream.traces[0].stats.endtime
            for pick in pick_list:
                phase = pick.phase_hint
                if starttime < pick.time < endtime:
                    if not tmp_pick.get(phase):
                        tmp_pick[phase] = [pick]
                    else:
                        tmp_pick[phase].append(pick)

            return tmp_pick

        def stream_get_pick(data):
            key, dictionary = data
            pick_list = dictionary['pick']
            stream_list = dictionary['stream']
            location = dictionary['location'][0]

            for stream in stream_list:
                picks = search_pick(pick_list, stream)
                stream.picks = picks
                stream.location = location
                yield stream

        return (
                pcollection
                | 'Stream search picks' >> beam.FlatMap(stream_get_pick)
        )


class GeneratePDF(beam.DoFn):
    def __init__(self, sigma=0.1):
        super(GeneratePDF, self).__init__()
        self.sigma = sigma

    def process(self, stream, *args, **kwargs):
        trace = stream[0]
        starttime = trace.stats.starttime
        x_time = trace.times(reftime=starttime)
        stream.pdf = {}

        for phase, picks in stream.picks.items():
            phase_pdf = np.zeros((len(x_time),))
            for pick in picks:
                pick_time = pick.time - starttime
                pick_pdf = ss.norm.pdf(x_time, pick_time, self.sigma)

                if pick_pdf.max():
                    phase_pdf += pick_pdf / pick_pdf.max()

            if phase_pdf.max():
                phase_pdf = phase_pdf / phase_pdf.max()

            stream.pdf[phase] = phase_pdf
        return [stream]
