from seisnn.db import Client


db = Client('HL2017.db')

db.generate_training_data('HL2017')
