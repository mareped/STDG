import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import DataConfig


def define_dict(classifier_type):
    hyperparameters = {
        logreg: {
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'penalty': ['l2'],
            'C': [10, 1.0, 0.1, 0.01]
        },
        mlp_classifier: {
            'solver': ['lbfgs', 'adam'],
            'activation': ['logistic', 'tanh', 'relu'],
            'alpha': 10.0 ** -np.arange(3, 7),
            'hidden_layer_sizes': np.arange(10, 15),
            'max_iter': [1000, 1200, 1400, 1600, 1800, 2000]
        }
    }

    if classifier_type not in hyperparameters:
        raise Exception("No such classifier exists")

    return hyperparameters[classifier_type]


def perform_randomized_search(dataset_path, classifier_type):
    dataset = pd.read_csv(dataset_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    hyperparameters = define_dict(classifier_type)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    randomized_search = RandomizedSearchCV(
        estimator=classifier_type, param_distributions=hyperparameters, n_iter=100,
        n_jobs=-1, cv=cv, scoring='accuracy', error_score=0, random_state=1
    )
    randomized_search_result = randomized_search.fit(X, y)
    print("Best: %f using %s" % (randomized_search_result.best_score_, randomized_search_result.best_params_))
    means = randomized_search_result.cv_results_['mean_test_score']
    stds = randomized_search_result.cv_results_['std_test_score']
    params = randomized_search_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def perform_grid_search(dataset_path, classifier_type):
    dataset = pd.read_csv(dataset_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    grid = define_dict(classifier_type)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=classifier_type, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                               error_score=0)
    grid_result = grid_search.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# create a new instance of DataConfig, with the dataset (lower_back_pain, obesity)
config = DataConfig(dataset_name='lower_back_pain', model_name='ctgan', epochs=800, batch_size=100)

real_path = config.real_path

mlp_classifier = MLPClassifier()
logreg = LogisticRegression()

# perform_randomized_search(real_path, mlp_classifier)

perform_grid_search(real_path, logreg)
