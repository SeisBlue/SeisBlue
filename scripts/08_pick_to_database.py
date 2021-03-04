import seisnn

dataset = 'eval'
database = 'HL2019.db'

dataset = seisnn.io.read_dataset(dataset)
db = seisnn.sql.Client(database)
for item in dataset:
    instance = seisnn.core.Instance(item)
    instance.predict.get_picks()
    instance.predict.write_picks_to_database('predict', database)
db.remove_duplicates('pick', ['time', 'phase', 'station', 'tag'])

