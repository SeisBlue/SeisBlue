import seisnn

test_set = 'HL2019'
model = 'test_model.h5'
database = 'HL2019.db'

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model)
evaluator.eval(test_set)
