import seisnn

database = 'HL2019.db'
db = seisnn.sql.Client(database)

tfr_list = [waveform.tfrecord for waveform in db.get_waveform()]

model_instance = 'test_model'
trainer = seisnn.model.trainer.GeneratorTrainer(database)
trainer.train_loop(tfr_list, model_instance, plot=True)
