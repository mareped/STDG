import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, f1_score, roc_curve
from utilities.plots import plot_multiclass_roc, plot_roc_binaryclass

# define dataset name you are using (lower_back_pain, obesity)
from sklearn.preprocessing import label_binarize

DATASET_NAME = 'lower_back_pain'

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


def read_data(dataset_path):
    dataset = pd.read_csv(dataset_path)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return x, y


def split_data(dataset_path):
    x, y = read_data(dataset_path)

    global classes
    classes = np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

    return X_train, X_test, y_train, y_test


def evaluate_classifier(clf, dataset1, dataset2):
    X_train, X_test, y_train, y_test = split_data(dataset1)
    x_2, y_2 = read_data(dataset2)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(x_2)

    y_pred_probs = clf.predict_proba(x_2)[:, 1]  # Predicted probabilities of the positive class

    f1 = f1_score(y_2, y_pred, average='weighted')

    return f1, y_2, y_pred_probs


def print_results(real_data, synth_data, mixed_data):
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=42),
        "Logistic Regression": LogisticRegression(C=100, penalty='l2', solver='newton-cg'),
        "MLP Classifier": MLPClassifier(max_iter=300, activation='relu', solver='adam')
    }

    results = []
    train_data = None
    test_data1 = None
    test_data2 = None

    fig, axes = plt.subplots(len(classifiers), 3, figsize=(15, 5 * len(classifiers)), squeeze=False)

    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        for jdx, (data_name, data) in enumerate({"real": real_data, "synth": synth_data, "mixed": mixed_data}.items()):

            ax = axes[idx, jdx]

            if data_name == "real":
                train_data = real_data
                test_data1 = real_data
                test_data2 = synth_data
            elif data_name == "synth":
                train_data = synth_data
                test_data1 = real_data
                test_data2 = synth_data
            elif data_name == "mixed":
                train_data = mixed_data
                test_data1 = real_data
                test_data2 = mixed_data

            if train_data is not None:
                train_test1_f1, y_true, y_score = evaluate_classifier(clf, train_data, test_data1)
                train_test2_f1, y_true_2, y_score_2 = evaluate_classifier(clf, train_data, test_data2)

                results.append({
                    'classifier': clf_name,
                    'train_data': data_name,
                    'f1_real': train_test1_f1,
                    'f1_synth': train_test2_f1
                })

                plot_roc_binaryclass(y_true, y_score, y_true_2, y_score_2, f'{clf_name}_{data_name}', ax=ax)

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(results)
    return df


print_results(real_path, fake_path, mixed_path)

"""global classes
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
_, y_true, y_score = evaluate_classifier(clf, real_path, fake_path)
y_true = label_binarize(y_true, classes=classes)

plot_multiclass_roc(y_true, y_score)"""
