import numpy as np
import pandas as pd
import tensorflow as tf

from seisnn.utils import unet_padding_size
from seisnn.plot import plot_dataset
from seisnn.io import write_tfrecord
from seisnn.pick import get_picks_from_pdf
from seisnn.example_proto import extract_parsed_example, feature_to_example


class Feature:
    def __init__(self, input_data=None):
        self.id = None
        self.station = None

        self.starttime = None
        self.endtime = None
        self.npts = None
        self.delta = None

        self.latitude = None
        self.longitude = None
        self.elevation = None

        self.channel = None
        self.picks = None
        self.phase = None

        self.trace = None
        self.pdf = None

        if isinstance(input_data['id'], str):
            self.from_feature(input_data)
        if isinstance(input_data['id'], tf.Tensor):
            self.from_example(input_data)

    def from_feature(self, feature):
        self.id = feature['id']
        self.station = feature['station']

        self.starttime = feature['starttime']
        self.endtime = feature['endtime']
        self.npts = feature['npts']
        self.delta = feature['delta']

        self.latitude = feature['latitude']
        self.longitude = feature['longitude']
        self.elevation = feature['elevation']

        self.channel = feature['channel']
        self.picks = feature['picks']
        self.phase = feature['phase']

    def to_feature(self):
        feature = {
            'id': self.id,
            'station': self.station,
            'starttime': self.starttime,
            'endtime': self.endtime,
            'npts': self.npts,
            'delta': self.delta,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'elevation': self.elevation,
            'channel': self.channel,
            'picks': self.picks,
            'phase': self.phase
        }
        return feature

    def from_example(self, example):
        feature = extract_parsed_example(example)
        self.from_feature(feature)

    def to_example(self):
        feature = self.to_feature()
        example = feature_to_example(feature)
        return example

    def to_tfrecord(self, file_path):
        feature = self.to_feature()
        example = feature_to_example(feature)
        write_tfrecord([example], file_path)

    def filter_phase(self, phase):
        keys = list(self.phase.keys())
        for key in keys:
            if not phase in key:
                self.phase.pop(key)

        self.picks = self.picks.loc[self.picks['pick_phase'] == phase]

    def filter_channel(self, channel):
        keys = list(self.channel.keys())
        for key in keys:
            if not channel in key:
                self.channel.pop(key)

    def filter_pickset(self, pickset):
        self.picks = self.picks.loc[self.picks['pick_set'] in pickset]

    def get_trace(self):
        traces = []
        for k, v in self.channel.items():
            tr = np.pad(v, unet_padding_size(v))
            traces.append(tr[np.newaxis, np.newaxis, :])
        if traces:
            traces = np.stack(traces, axis=-1)
            self.trace = traces
            return traces
        else:
            return None

    def get_pdf(self):
        pdf = []
        for k, v in self.phase.items():
            tr = np.pad(v, unet_padding_size(v))
            pdf.append(tr[np.newaxis, np.newaxis, :])
        if pdf:
            pdf = np.stack(pdf, axis=-1)
            self.pdf = pdf
            return pdf
        else:
            return None

    def get_picks(self, phase_type):
        picks = get_picks_from_pdf(self, phase_type)
        self.picks = pd.concat([self.picks, picks])

    def plot(self, **kwargs):
        feature = self.to_feature()
        plot_dataset(feature, **kwargs)
