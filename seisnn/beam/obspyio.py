import os

import apache_beam as beam
import numpy as np
import scipy
from obspy.clients.filesystem.sds import Client
from obspy.io.nordic.core import read_nordic


class PickFeatureExtraction(beam.DoFn):
    def process(self, pick, *args, **kwargs):
        feature = {
            'pick_time': pick.time,
            'pick_phase': pick.phase_hint,
            'pick_type': pick.evaluation_mode,

            'station': pick.waveform_id.station_code
        }
        return [feature]


class TraceFeatureExtraction(beam.DoFn):
    def process(self, trace, *args, **kwargs):
        feature = {
            'starttime': trace.stats.starttime,
            'endtime': trace.stats.endtime,

            'network': trace.stats.network,
            'location': trace.stats.location,
            'station': trace.stats.station,
            'channel': trace.stats.channel,

            'npts': trace.stats.npts,
            'delta': trace.stats.delta,

            'data': trace.data
        }
        return [feature]


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
        return stream


class TrimTrace(beam.DoFn):
    def __init__(self, points=3001):
        super(TrimTrace, self).__init__()
        self.points = points

    def process(self, trace, *args, **kwargs):
        start_time = trace.stats.starttime
        dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
        end_time = start_time + dt * (self.points - 1)
        trace.trim(start_time, end_time, nearest_sample=False, pad=True, fill_value=0)
        return [trace]



class FilterPickPhase(beam.DoFn):
    def __init__(self, phase):
        super(FilterPickPhase, self).__init__()
        self.phase = phase

    def process(self, pick, *args, **kwargs):
        if pick.phase_hint == self.phase:
            return [pick]
        else:
            return
