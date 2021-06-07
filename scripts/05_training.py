import seisnn

database = 'CWB.db'
db = seisnn.sql.Client(database)

tfr_list = db.get_tfrecord(to_date='2019-05-09',column='path')

model_instance = 'test_model'
trainer = seisnn.model.trainer.GeneratorTrainer(database)
trainer.train_loop(tfr_list, model_instance,
                   batch_size=64, epochs=10,
                   plot=True)
