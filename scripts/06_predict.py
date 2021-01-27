from seisnn.model.trainer import GeneratorTrainer
from seisnn.model.evaluator import GeneratorEvaluator

test_set = 'HL2018'
model_instance = 'test_model'
database = 'HL2017.db'

trainer = GeneratorTrainer(database)
trainer.export_model(model_instance)

evaluator = GeneratorEvaluator(database, model_instance)
evaluator.eval(test_set)
