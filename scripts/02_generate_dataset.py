import seisblue
import tensorflow as tf

print(tf.constant(0))
database = 'demo'
tag = 'manual'

db = seisblue.sql.Client(database=database)
inspector = seisblue.sql.DatabaseInspector(db)

pick_list = db.get_picks(tag=tag, from_time='2020-04-01', to_time='2020-04-10',
                         station='HP*')

tfr_converter = seisblue.components.TFRecordConverter(trace_length=33,
                                                      phase=('P', 'S', 'N'))
tfr_converter.convert_training_from_picks(pick_list, tag, database)

config = seisblue.utils.Config()
tfr_list = seisblue.utils.get_dir_list(config.train, suffix='.tfrecord')

db.clear_table(table='waveform')
db.clear_table(table='tfrecord')
db.read_tfrecord_header(tfr_list)
inspector.waveform_summery()
