import seisnn

model_instance = 'test_model'
database = 'Hualien.db'
tfr_list = seisnn.utils.get_dir_list('/home/andy/TFRecord/Eval/QQQ.h5/', suffix='.tfrecord')

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model_instance)
evaluator.score(tfr_list, height=0.5,delta=0.1, error_distribution=True)


