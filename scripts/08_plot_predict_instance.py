from seisnn import data

dataset = 'eval'

dataset = data.io.read_dataset(dataset)
for item in dataset:
    instance = data.core.Instance(item)
    instance.plot()
