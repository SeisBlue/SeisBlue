import tensorflow as tf

from seisnn.plot import plot_dataset
from seisnn.io import write_tfrecord
from seisnn.pick import get_picks_from_pdf
from seisnn.example_proto import eval_eager_tensor, feature_to_example


class Feature:
    def __init__(self, example):
        self.id = None
        self.station = None

        self.starttime = None
        self.endtime = None
        self.npts = None
        self.delta = None

        self.latitude = None
        self.longitude = None
        self.elevation = None

        self.trace = None
        self.channel = None

        self.phase = None
        self.pdf = None

        self.pick_time = None
        self.pick_phase = None
        self.pick_set = None

        self.from_example(example)


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

        self.trace = feature['trace']
        self.channel = feature['channel']

        self.pdf = feature['pdf']
        self.phase = feature['phase']

        self.pick_time = feature['pick_time']
        self.pick_phase = feature['pick_phase']
        self.pick_set = feature['pick_set']


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

            'trace': self.trace,
            'channel': self.channel,

            'phase': self.phase,
            'pdf': self.pdf,

            'pick_time': self.pick_time,
            'pick_phase': self.pick_phase,
            'pick_set': self.pick_set,

        }
        return feature

    def from_example(self, example):
        feature = eval_eager_tensor(example)
        self.from_feature(feature)

    def to_example(self):
        feature = self.to_feature()
        example = feature_to_example(feature)
        return example

    def to_tfrecord(self, file_path):
        feature = self.to_feature()
        example = feature_to_example(feature)
        write_tfrecord([example], file_path)

    def get_picks(self, phase, pick_set):
        get_picks_from_pdf(self, phase, pick_set)

    def plot(self, **kwargs):
        feature = self.to_feature()
        plot_dataset(feature, **kwargs)


def parallel_to_tfrecord(batch_list):
    from seisnn.utils import parallel

    example_list = parallel(par=_to_tfrecord, file_list=batch_list)
    return example_list

def _to_tfrecord(batch):
    example_list=[]
    for example in batch:
        feature = Feature(example)
        feature.get_picks('p', 'predict')
        feature = feature.to_feature()
        example_list.append(feature_to_example(feature))
    return example_list