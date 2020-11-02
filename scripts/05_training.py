from seisnn.model.trainer import GeneratorTrainer

dataset = 'HL2017'
model_instance = 'test_model'
database = 'HL2017.db'

trainer = GeneratorTrainer(database)
trainer.train_loop(dataset, model_instance, plot=True)
