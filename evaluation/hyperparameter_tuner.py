import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# define dataset name you are using (lower_back_pain, obesity)
DATASET_NAME = 'lower_back_pain'

# define dataset model you are using (ctgan, copulagan)
MODEL_NAME = 'ctgan'

# Hyperparameters
EPOCHS = 400
BATCH_SIZE = 100

file_ending = f'{MODEL_NAME}_{EPOCHS}_epochs_{BATCH_SIZE}_batch'

# Define where the real and fake data path is. IMPORTANT: change real file name
real_path = f'../data/{DATASET_NAME}/{DATASET_NAME}_scaled.csv'
fake_path = f'../data/{DATASET_NAME}/' + file_ending + '.csv'
mixed_path = f'../data/{DATASET_NAME}/' + file_ending + '.csv'

mlp_classifier = MLPClassifier()
logreg = LogisticRegression()


def define_dict(classifier_type):
    if classifier_type == logreg:
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        grid = dict(solver=solvers, penalty=penalty, C=c_values)

    elif classifier_type == mlp_classifier:
        solvers = ['lbfgs', 'sgd', 'adam']
        activations = ['logistic', 'tanh', 'relu']
        alphas = 10.0 ** -np.arange(1, 7)
        hidden_layer_size = np.arange(10, 15)
        max_iters= [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

        grid = dict(
            max_iter = max_iters, solver=solvers, alpha=alphas, activation=activations,
            hidden_layer_sizes=hidden_layer_size)
    else:
        raise Exception("No such classifier exist")

    return grid


def perform_grid_search(dataset_path, classifier_type):
    dataset = pd.read_csv(dataset_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    grid = define_dict(classifier_type)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=classifier_type, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


perform_grid_search(real_path, mlp_classifier)
