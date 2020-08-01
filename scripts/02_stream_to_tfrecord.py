import seisnn

dataset = 'HL2017'
db = seisnn.data.sql.Client('test.db')
db.pick_summery()

db.generate_training_data(dataset)
db.sync_dataset_header(dataset)
db.waveform_summery()
