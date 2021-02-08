import seisnn

model_instance = 'test_model'
database = 'HL2019.db'

trainer = seisnn.model.trainer.GeneratorTrainer(database)
trainer.export_model(model_instance)
