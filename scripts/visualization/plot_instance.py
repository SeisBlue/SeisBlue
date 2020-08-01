import os

import seisnn

dataset = 'HL2017'

config = seisnn.utils.get_config()
dataset_dir = os.path.join(config['TFRECORD_ROOT'], dataset)
dataset = seisnn.data.io.read_dataset(dataset_dir)

for batch in dataset.shuffle(1000).batch(2):
    for example in seisnn.data.example_proto.batch_iterator(batch):
        feature = seisnn.data.core.Instance(example)
        feature.plot(enlarge=True, snr=True)
