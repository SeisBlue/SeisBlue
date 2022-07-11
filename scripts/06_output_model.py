import seisblue

trainer = seisblue.model.trainer.GeneratorTrainer(database='Hualien.db')
trainer.export_model(model_name='test_model')
