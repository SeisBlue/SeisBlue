import seisblue

# model_instance = 'CWB_2010_2019_trans'
# database = 'CWB.db'
tfr_list = seisblue.utils.get_dir_list('/home/andy/TFRecord/Eval/demo_gan.h5',
                                       suffix='.tfrecord')
evaluator = seisblue.model.evaluator.GeneratorEvaluator()

evaluator.score(tfr_list, threshold=0.3, delta=0.2, error_distribution=True)
