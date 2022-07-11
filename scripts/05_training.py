import seisblue

database = 'demo'
db = seisblue.sql.Client(database)

tfr_list = db.get_tfrecord(from_date='2020-04-01', column='path')
tfr_list = seisblue.utils.flatten_list(tfr_list)
noise = seisblue.utils.get_dir_list('/home/jimmy/TFRecord/noise',
                                    suffix='.tfrecord')
for item in noise:
    tfr_list.append(item)
model_instance = 'demo_model'
trainer = seisblue.model.trainer.GeneratorTrainer(database)
trainer.train_loop(tfr_list, model_instance,
                   batch_size=250, epochs=100, log_step=200,
                   plot=True)
