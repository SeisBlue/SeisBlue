import seisnn

model = 'test_model.h5'
database = 'Hualien.db'

db = seisnn.sql.Client(database)
tfr_list = db.get_matched_list('*', 'tfrecord', 'path')

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model)
evaluator.eval(tfr_list)
