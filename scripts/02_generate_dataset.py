import seisnn

database = 'Hualien.db'
tag = 'manual'

db = seisnn.sql.Client(database=database)
inspector = seisnn.sql.DatabaseInspector(db)
inspector.pick_summery()

pick_list = db.get_picks(tag=tag).all()

tfr_converter = seisnn.components.TFRecordConverter()
tfr_converter.convert_training_from_picks(pick_list, tag, database)

config = seisnn.utils.Config()
tfr_list = seisnn.utils.get_dir_list(config.train, suffix='.tfrecord')

db.clear_table(table='waveform')
db.clear_table(table='tfrecord')
db.read_tfrecord_header(tfr_list)
inspector.waveform_summery()
