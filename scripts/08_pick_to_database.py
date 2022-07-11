import seisblue

config = seisblue.utils.Config()

database = 'Hualien.db'
db = seisblue.sql.Client(database)

tfr_list = seisblue.utils.get_dir_list(config.tfrecord, suffix='.tfrecord')
dataset = seisblue.io.read_dataset(tfr_list)

for item in dataset:
    instance = seisblue.core.Instance(item)
    instance.predict.get_picks()
    instance.predict.write_picks_to_database('predict', database)
db.remove_duplicates('pick', ['time', 'phase', 'station', 'tag'])
