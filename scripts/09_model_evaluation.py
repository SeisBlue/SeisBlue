import seisnn

model_instance = 'test_model'
database = 'Hualien.db'
evaluator = seisnn.model.evaluator.GeneratorEvaluator(database, model_instance)
tfr_list = evaluator.get_eval_list()
evaluator.score(tfr_list, delta=0.5, error_distribution=True)

