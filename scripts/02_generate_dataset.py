import seisnn

database = 'Hualien.db'
tag = 'manual'

db = seisnn.sql.Client(database=database)
db.pick_summery()

pick_list = db.get_picks(tag=tag).all()

tfr_converter = seisnn.components.TFRecordConverter()
tfr_converter.convert_from_picks(pick_list, tag, database)

config = seisnn.utils.Config()
tfr_list = seisnn.utils.get_dir_list(config.tfrecord, suffix='.tfrecord')

db.clear_table(table='waveform')
db.clear_table(table='tfrecord')
db.read_tfrecord_header(tfr_list)
db.waveform_summery()
