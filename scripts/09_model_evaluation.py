from seisnn.model.evaluator import GeneratorEvaluator

test_set = 'HL2017'
model_instance = 'test_model'
database = 'HL2017.db'

evaluator = GeneratorEvaluator(database, model_instance)
