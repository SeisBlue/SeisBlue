import seisnn

model = 'test_model.h5'
database = 'HL2019.db'

db = seisnn.sql.Client(database)
tfr_list = [waveform.tfrecord for waveform in db.get_waveform()]

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model)
evaluator.eval(tfr_list, 'eval')
