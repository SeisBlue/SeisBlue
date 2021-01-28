from seisnn.model.trainer import GeneratorTrainer

model_instance = 'test_model'
database = 'HL2019.db'

trainer = GeneratorTrainer(database)
trainer.export_model(model_instance)