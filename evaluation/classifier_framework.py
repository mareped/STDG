import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, f1_score, roc_curve

# define dataset name you are using (lower_back_pain, obesity)
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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

    return X_train, X_test, y_train, y_test


def evaluate_classifier(clf, dataset_r, dataset_f, classifier="real", ax=None):
    x_r, y_r = read_data(dataset_r)
    x_f, y_f = read_data(dataset_f)

    X_train_r, X_test_r, y_train_r, y_test_r = split_data(dataset_r)
    X_train_f, X_test_f, y_train_f, y_test_f = split_data(dataset_f)

    if classifier == "real":
        X_train, X_test, y_train, y_test = X_train_r, X_test_r, y_train_r, y_test_r
        x_other, y_other = x_f, y_f
        label_other = "fake"
    elif classifier == "fake":
        X_train, X_test, y_train, y_test = X_train_f, X_test_f, y_train_f, y_test_f
        x_other, y_other = x_r, y_r
        label_other = "real"
    else:
        raise Exception("Need to specify real or fake classifier")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_other = clf.predict(x_other)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_proba_other = clf.predict_proba(x_other)[:, 1]


    f1 = f1_score(y_test.ravel(), y_pred, average='weighted')
    f1_2 = f1_score(y_other.ravel(), y_pred_other, average='weighted')
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred_proba.ravel())
    fpr_other, tpr_other, _ = roc_curve(y_other.ravel(), y_pred_proba_other.ravel())

    auc = roc_auc_score(y_test, y_pred_proba)
    auc_2 = roc_auc_score(y_other, y_pred_proba_other)

    ax.plot(fpr, tpr, label=classifier.capitalize())
    ax.plot(fpr_other, tpr_other, label=label_other)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{type(clf).__name__}_{classifier}')
    ax.legend()

    results = {
        'classifier': type(clf).__name__ + "_" + classifier,
        f'f1_{classifier}': f1,
        f'f1_{label_other}': f1_2,
        f'auc_{classifier}': auc,
        f'auc_{label_other}': auc_2
    }
    return results


def create_results(dataset_real, dataset_fake):
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    logreg = LogisticRegression(C=100, penalty='l2', solver='newton-cg')
    mlp = MLPClassifier(max_iter=300, activation='relu', solver='adam')

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    classifiers = []
    for i, (clf, dataset_r, dataset_f, classifier) in enumerate([(logreg, dataset_real, dataset_fake, "real"),
                                                                 (logreg, dataset_real, dataset_fake, "fake"),
                                                                 (mlp, dataset_real, dataset_fake, "real"),
                                                                 (mlp, dataset_real, dataset_fake, "fake"),
                                                                 (rf, dataset_real, dataset_fake, "real"),
                                                                 (rf, dataset_real, dataset_fake, "fake")]):
        row = i // 3
        col = i % 3
        result = evaluate_classifier(clf, dataset_r, dataset_f, classifier=classifier, ax=axes[row, col])
        classifiers.append(result)

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame.from_records(classifiers, columns=['classifier', 'f1_real', 'f1_fake', 'auc_real', 'auc_fake'])
    print(df)


create_results(real_path, fake_path)
