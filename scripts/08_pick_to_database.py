from seisnn import data

dataset = 'eval'
dataset = data.io.read_dataset(dataset)
for item in dataset:
    instance = data.core.Instance(item)
    instance.predict_into_database(tag = 'predict',database = "HL2019.db")

