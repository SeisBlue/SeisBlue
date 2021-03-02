import seisnn

test_set = 'HL2019test'
model_instance = 'test_model'
database = 'HL2019.db'

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model_instance)
evaluator.score(delta=0.1,error_distribution=True)
