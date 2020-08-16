from seisnn.model.core import Trainer

dataset = 'HL2017'
model_instance = 'test_model'

trainer = Trainer()
trainer.train_loop(dataset, model_instance)
