from config import DataConfig
from evaluation.synth_eval_framework import SynthEval

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


config = DataConfig(dataset_name='obesity', model_name='ctgan', epochs=300, batch_size=50)

real_path, fake_path, result_path = config.real_path, config.fake_path, config.result_path

# declare which classifiers to use
logreg = LogisticRegression()
rf = RandomForestClassifier()
mlp = MLPClassifier()

evaluator = SynthEval(real_path, fake_path, result_path)

evaluator.add_all_classifiers(logreg, rf, mlp)

evaluator.compare_datasets_performance(real_percentage=0.5, synth_percentage=1, cross_val=True)

