import seisnn

config = seisnn.utils.get_config()

dataset = seisnn.io.read_dataset(dataset)
for item in dataset:
    instance = seisnn.core.Instance(item)
    instance.plot()
