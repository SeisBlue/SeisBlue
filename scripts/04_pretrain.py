import seisnn

database = 'Hualien.db'
db = seisnn.sql.Client(database=database)

waveform_list = db.get_waveform()

model_instance = 'test_model'
trainer = seisnn.model.trainer.GeneratorTrainer(database=database)
trainer.train_loop(waveform_list, model_instance, plot=True, remove=True)
