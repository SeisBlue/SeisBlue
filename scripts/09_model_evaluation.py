import seisnn.model

test_set = 'HL2017'
model_instance = 'test_model'
database = 'HL2017.db'

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model_instance)
