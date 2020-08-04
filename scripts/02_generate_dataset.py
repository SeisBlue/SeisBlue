import seisnn

dataset = 'HL2017'
db = seisnn.data.sql.Client('HL2017.db')
db.pick_summery()
db.get_picks()

db.generate_training_data(dataset)

db.clear_table('waveform')
db.read_tfrecord_header(dataset)
db.waveform_summery()
