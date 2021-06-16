import seisnn

model = 'CWB_2010_2019_transgan.h5'
database = 'CWB.db'

db = seisnn.sql.Client(database)
tfr_list = db.get_tfrecord(from_date='2020-01-01',to_date='2020-12-31',column='path')
tfr_list = seisnn.utils.flatten_list(tfr_list)

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model)
evaluator.predict(tfr_list)
