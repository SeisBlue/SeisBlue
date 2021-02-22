import seisnn

dataset = 'eval'

dataset = seisnn.io.read_dataset(dataset)
for item in dataset:
    instance = seisnn.core.Instance(item)
    instance.plot()
