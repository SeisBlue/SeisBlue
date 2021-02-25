import seisnn

dataset = 'eval'
dataset = seisnn.io.read_dataset(dataset)

for item in dataset:
    instance = seisnn.core.Instance(item)
    seisnn.processing.get_picks_from_predict(instance,
                                             tag='val_predict',
                                             database="HL2019.db")
    seisnn.processing.get_picks_from_manual(instance,
                                             tag='val_manual',
                                             database="HL2019.db")
