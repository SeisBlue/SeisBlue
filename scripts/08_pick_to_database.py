import seisnn

dataset = 'eval'
dataset = seisnn.io.read_dataset(dataset)
for item in dataset:
    instance = seisnn.core.Instance(item)
    instance.predict_into_database(tag='predict', database="HL2019.db")
