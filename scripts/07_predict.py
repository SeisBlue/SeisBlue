import seisnn

model = 'QQQQ.h5'
database = 'Hualien.db'

db = seisnn.sql.Client(database)
tfr_list = db.get_tfrecord(from_date='2019-05-09',column='path')

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model)
evaluator.predict(tfr_list)
