import seisblue

model = 'demo_gan.h5'
database = 'demo'

db = seisblue.sql.Client(database)
tfr_list = db.get_tfrecord(from_date='2020-04-01', to_date='2020-06-30',
                           column='path')
tfr_list = seisblue.utils.flatten_list(tfr_list)

# tfr_list = seisblue.utils.get_dir_list('/home/andy/TFRecord/noise/HP01',suffix='.tfrecord')

evaluator = seisblue.model.evaluator.GeneratorEvaluator(database, model)
evaluator.predict(tfr_list)
