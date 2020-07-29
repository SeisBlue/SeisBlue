from seisnn import sql

db = sql.Client('test.db')
db.pick_summery()
result = db.get_picks().all()
print(result)


db.generate_training_data('HL2017')
