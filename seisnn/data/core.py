"""
Core
"""

from seisnn import data
from seisnn import plot
from seisnn import processing


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
    pdf = None

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

    def __repr__(self):
        return f"Instance(" \
               f"ID={self.id}, " \
               f"Start Time={self.starttime}, " \
               f"Phase={self.phase})"

    def from_feature(self, feature):
        """
        Initialized from feature dict.

        :type feature: dict
        :param feature: Feature dict.
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
        Returns feature dict.

        :rtype: dict
        :return: Feature dict.
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
        Initialized from example protocol.

        :param example: Example protocol.
        """
        feature = data.example_proto.eval_eager_tensor(example)
        self.from_feature(feature)

    def to_example(self):
        """
        Returns example protocol.

        :return: Example protocol.
        """
        feature = self.to_feature()
        example = data.example_proto.feature_to_example(feature)
        return example

    def to_tfrecord(self, file_path):
        """
        Write TFRecord to file path.

        :param str file_path: Output path.
        """
        feature = self.to_feature()
        example = data.example_proto.feature_to_example(feature)
        data.io.write_tfrecord([example], file_path)

    def get_picks(self, phase, pick_set):
        """
        Extract picks from pdf.

        :param str phase: Phase name.
        :param str pick_set: Pick set name.
        """
        processing.get_picks_from_pdf(self, phase, pick_set)

    def plot(self, **kwargs):
        """
        Plot dataset.

        :param kwargs: Keywords pass into plot.
        """
        feature = self.to_feature()
        plot.plot_dataset(feature, **kwargs)


def parallel_to_tfrecord(batch_list):
    """
    Writed TFRecord to directory.

    :param batch_list:
    :return: List of results.
    """
    from seisnn.utils import parallel

    example_list = parallel(par=_to_tfrecord, file_list=batch_list)
    return example_list


def _to_tfrecord(batch):
    """
    Returns example list from batched example.

    :param batch: Batch list of example.
    """
    example_list = []
    for example in batch:
        instance = Instance(example)
        instance.get_picks('p', 'predict')
        feature = instance.to_feature()
        example_list.append(data.example_proto.feature_to_example(feature))
    return example_list


if __name__ == "__main__":
    pass
