from config import DataConfig
from evaluation.clf_framework import ClassifierFramework

config = DataConfig(dataset_name='obesity', model_name='copulagan', epochs=350, batch_size=100)

# real_path, fake_path, mixed_path, result_path = config.real_path, config.fake_path, config.mixed_path, config.result_path
real_path, fake_path, mixed_path, result_path = \
    f'{config.real_path}_prep.csv', f'{config.fake_path}_prep.csv', f'{config.mixed_path}_prep.csv', config.result_path

evaluator = ClassifierFramework()
evaluator.print_results(real_path, fake_path, mixed_path, result_path)
