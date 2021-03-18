import seisnn

database = 'Hualien.db'
db = seisnn.sql.Client(database=database)

tfr_list = db.get_matched_list('*', 'tfrecord', 'path')

model_instance = 'test_model'
trainer = seisnn.model.trainer.GeneratorTrainer(database=database)
trainer.train_loop(tfr_list, model_instance,
                   batch_size=64,
                   plot=True, remove=True)
