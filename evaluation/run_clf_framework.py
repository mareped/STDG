from config import DataConfig
from evaluation.classifer_eval_framework import ClassifierEvaluationFramework

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


config = DataConfig(dataset_name='lower_back_pain', model_name='ctgan', epochs=600, batch_size=100)

real_path, fake_path, mixed_path, result_path = config.real_path, config.fake_path, config.mixed_path, config.result_path

# declare which classifiers to use
logreg = LogisticRegression()
rf = RandomForestClassifier()
mlp = MLPClassifier()
#dt = DecisionTreeClassifier()

evaluator = ClassifierEvaluationFramework()

evaluator.add_all_classifiers(logreg, rf, mlp)

evaluator.t1t2_results(real_path, fake_path, mixed_path, result_path, test_size=0.2)

