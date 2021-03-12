import seisnn

database = 'Hualien.db'
tag = 'manual'

db = seisnn.sql.Client(database=database)
db.pick_summery()

pick_list = db.get_picks(tag=tag).all()

example_gen = seisnn.components.ExampleGen()
example_gen.generate_training_data(pick_list, dataset, tag, database)

db.clear_table('waveform')
db.read_tfrecord_header(dataset)
db.waveform_summery()
