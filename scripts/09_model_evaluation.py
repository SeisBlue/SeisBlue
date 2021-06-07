import seisnn

model_instance = 'CWB_2010_2019_trans'
database = 'CWB.db'
tfr_list = seisnn.utils.get_dir_list('/home/andy/TFRecord/Eval/CWB_2010_2019_transgan.h5/', suffix='.tfrecord')

evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model_instance)
evaluator.score(tfr_list, height=0.4,delta=0.5, error_distribution=True)


