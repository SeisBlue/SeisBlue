"""
Core
=============

.. autosummary::
    :toctree: core

    Feature
    parallel_to_tfrecord

"""

from seisnn import example_proto
from seisnn import io
from seisnn import plot
from seisnn import processing


class Feature:
    def __init__(self, example):
        self.id = None
        self.station = None

        self.starttime = None
        self.endtime = None
        self.npts = None
        self.delta = None

        self.trace = None
        self.channel = None

        self.phase = None
        self.pdf = None

        self.from_example(example)

    def from_feature(self, feature):
        """

        :param feature:
        """
        self.id = feature['id']
        self.station = feature['station']

        self.starttime = feature['starttime']
        self.endtime = feature['endtime']
        self.npts = feature['npts']
        self.delta = feature['delta']

        self.trace = feature['trace']
        self.channel = feature['channel']

        self.pdf = feature['pdf']
        self.phase = feature['phase']

    def to_feature(self):
        """

        :return:
        """
        feature = {
            'id': self.id,
            'station': self.station,
            'starttime': self.starttime,
            'endtime': self.endtime,

            'npts': self.npts,
            'delta': self.delta,

            'trace': self.trace,
            'channel': self.channel,

            'phase': self.phase,
            'pdf': self.pdf,
        }
        return feature

    def from_example(self, example):
        """

        :param example:
        """
        feature = example_proto.eval_eager_tensor(example)
        self.from_feature(feature)

    def to_example(self):
        """

        :return:
        """
        feature = self.to_feature()
        example = example_proto.feature_to_example(feature)
        return example

    def to_tfrecord(self, file_path):
        """

        :param file_path:
        """
        feature = self.to_feature()
        example = example_proto.feature_to_example(feature)
        io.write_tfrecord([example], file_path)

    def get_picks(self, phase, pick_set):
        """

        :param phase:
        :param pick_set:
        """
        processing.get_picks_from_pdf(self, phase, pick_set)

    def plot(self, **kwargs):
        """

        :param kwargs:
        """
        feature = self.to_feature()
        plot.plot_dataset(feature, **kwargs)


def parallel_to_tfrecord(batch_list):
    """

    :param batch_list:
    :return:
    """
    from seisnn.utils import parallel

    example_list = parallel(par=_to_tfrecord, file_list=batch_list)
    return example_list


def _to_tfrecord(batch):
    """
    """
    example_list = []
    for example in batch:
        feature = Feature(example)
        feature.get_picks('p', 'predict')
        feature = feature.to_feature()
        example_list.append(example_proto.feature_to_example(feature))
    return example_list


if __name__ == "__main__":
    pass
