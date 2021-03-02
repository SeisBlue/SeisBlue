import seisnn

dataset = 'eval'
dataset = seisnn.io.read_dataset(dataset)
db = seisnn.sql.Client('HL2019.db')
for item in dataset:
    instance = seisnn.core.Instance(item)
    seisnn.processing.get_picks_from_eval(instance,
                                          db=db)
db.remove_duplicates('pick', ['time', 'phase', 'station', 'tag'])

