import seisnn

trainer = seisnn.model.trainer.GeneratorTrainer(database='Hualien.db')
trainer.export_model(model_name='test_model')
