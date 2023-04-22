from config import DataConfig
from evaluation.clf_framework import ClassifierFramework

config = DataConfig(dataset_name='lower_back_pain', model_name='copulagan', epochs=800, batch_size=100)

real_path, fake_path, mixed_path, result_path = config.real_path, config.fake_path, config.mixed_path, config.result_path
#real_path, fake_path, mixed_path, result_path = \
#   f'{config.real_path}_encoded.csv', f'{config.fake_path}_encoded.csv', f'{config.mixed_path}_encoded.csv', config.result_path

evaluator = ClassifierFramework()
evaluator.print_t1t2_results(real_path, fake_path, mixed_path, result_path)

classifier = evaluator.classifiers.get("Random Forest")
#evaluator.plot_confusion_matrix(classifier, real_path)
