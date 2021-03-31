import seisnn

config = seisnn.utils.Config()

database = 'Hualien.db'
db = seisnn.sql.Client(database)

tfr_list = seisnn.utils.get_dir_list(config.tfrecord, suffix='.tfrecord')
dataset = seisnn.io.read_dataset(tfr_list)

for item in dataset:
    instance = seisnn.core.Instance(item)
    instance.predict.get_picks()
    instance.predict.write_picks_to_database('predict', database)
db.remove_duplicates('pick', ['time', 'phase', 'station', 'tag'])
