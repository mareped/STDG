from config import DataConfig
from evaluation.classifer_eval_framework import ClassifierEvaluationFramework

config = DataConfig(dataset_name='lower_back_pain', model_name='ctgan', epochs=600, batch_size=100)

real_path, fake_path, mixed_path, result_path = config.real_path, config.fake_path, config.mixed_path, config.result_path

evaluator = ClassifierEvaluationFramework()
evaluator.print_t1t2_results(real_path, fake_path, mixed_path, result_path, test_size=0.5)

#classifier = evaluator.classifiers.get("Random Forest")
#evaluator.plot_confusion_matrix(classifier, real_path)
