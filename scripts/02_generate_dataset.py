import seisnn

dataset = 'HL2019'
db = seisnn.data.sql.Client('HL2019.db')
db.pick_summery()

pick_list = db.get_picks(phase='S').all()
db.generate_training_data(pick_list, dataset)

db.clear_table('waveform')
db.read_tfrecord_header(dataset)
db.waveform_summery()
