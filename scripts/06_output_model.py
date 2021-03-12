import seisnn

trainer = seisnn.model.trainer.GeneratorTrainer(database='HL2019.db')
trainer.export_model(model_name='test_model')
