import seisblue

database = 'Hualien.db'
db = seisblue.sql.Client(database=database)

tfr_list = db.get_tfrecord(to_date='2019-04-20', column='path')

model_instance = 'test_model'
trainer = seisblue.model.trainer.GeneratorTrainer(database=database)
trainer.train_loop(tfr_list, model_instance,
                   batch_size=64,
                   plot=True, remove=True)
