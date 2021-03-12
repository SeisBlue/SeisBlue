import seisnn

config = seisnn.utils.Config()

tfr_list = seisnn.utils.get_dir_list(config.tfrecord, suffix='.tfrecord')

dataset = seisnn.io.read_dataset(tfr_list)
for item in dataset:
    instance = seisnn.core.Instance(item)
    instance.plot()
