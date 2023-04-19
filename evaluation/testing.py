import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, f1_score, roc_curve

# define dataset name you are using (lower_back_pain, obesity)
from sklearn.preprocessing import label_binarize
from utilities.plots import plot_multiclass_roc


DATASET_NAME = 'obesity'

# define dataset model you are using (ctgan, copulagan)
MODEL_NAME = 'ctgan'

# Hyperparameters
EPOCHS = 800
BATCH_SIZE = 100

file_ending = f'{MODEL_NAME}_{EPOCHS}_epochs_{BATCH_SIZE}_batch'

# Define where the real and fake data path is. IMPORTANT: change real file name
real_path = f'../data/{DATASET_NAME}/{DATASET_NAME}_scaled.csv'
fake_path = f'../data/{DATASET_NAME}/' + file_ending + '.csv'
mixed_path = f'../data/{DATASET_NAME}/' + file_ending + '.csv'

global classes


def read_data(dataset_path):
    dataset = pd.read_csv(dataset_path)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return x, y


def split_data(dataset_path):
    x, y = read_data(dataset_path)

    global classes
    classes = np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True, stratify=y, random_state=16)

    return X_train, X_test, y_train, y_test


def evaluate_classifier(clf, dataset1):
    X_train, X_test, y_train, y_test = split_data(dataset1)

    clf.fit(X_train, y_train)
    y_pred_probs = clf.predict_proba(X_test)  # Predicted probabilities for each class

    # Binarize the true labels
    y_true = label_binarize(y_test, classes=classes)

    return y_true, y_pred_probs

# example usage
clf = MLPClassifier()
y_true, y_score = evaluate_classifier(clf, real_path)
n_classes = len(classes)
print(n_classes)
plot_multiclass_roc(y_true, y_score, n_classes, False)
