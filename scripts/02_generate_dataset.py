import seisnn

dataset = 'HL2019'
database = 'HL2019.db'
tag = 'manual'

db = seisnn.sql.Client(database)
db.pick_summery()

pick_list = db.get_picks(phase='S').all()

example_gen = seisnn.component.ExampleGen()
example_gen.generate_training_data(pick_list, dataset, tag, database)

db.clear_table('waveform')
db.read_tfrecord_header(dataset)
db.waveform_summery()
