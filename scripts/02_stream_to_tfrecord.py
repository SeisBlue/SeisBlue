import seisnn

db = seisnn.data.sql.Client('test.db')
db.pick_summery()

db.generate_training_data('HL2017')
