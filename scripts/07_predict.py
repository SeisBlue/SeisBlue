import seisnn

model = 'test_model.h5'
database = 'HL2019.db'

db = seisnn.sql.Client(database)
tfr_list = db.get_matched_list("2019", 'tfrecord', 'path')

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model)
evaluator.eval(tfr_list, 'eval')
