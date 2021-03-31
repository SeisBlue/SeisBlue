import seisnn


database = 'Hualien.db'
db = seisnn.sql.Client(database)
testset = db.get_waveform()

model_instance = 'test_model'
evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model_instance)
evaluator.score(delta=0.1, error_distribution=True)
