import seisnn

model_instance = 'test_model'
database = 'Hualien.db'
evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model_instance)
tfr_list = evaluator.get_eval_list()
evaluator.score(tfr_list, height=0.5,delta=0.1, error_distribution=True)


