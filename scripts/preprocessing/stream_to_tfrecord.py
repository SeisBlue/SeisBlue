from seisnn.sql import Client, get_table_class

db = Client('HL2017.db')

db.generate_training_data('HL2017')
