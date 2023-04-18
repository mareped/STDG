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


def evaluate_classifier(clf, dataset1, dataset2):
    X_train, X_test, y_train, y_test = split_data(dataset1)
    x_2, y_2 = read_data(dataset2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(x_2)

    y_pred_probs = clf.predict_proba(x_2)[:, 1]  # Predicted probabilities of the positive class

    f1 = f1_score(y_2, y_pred, average='weighted')

    return f1, y_2, y_pred_probs


def plot_roc_binary(y_true_real, y_score_real, y_true_synth, y_score_synth, clf_name, ax=None):
    """Args:
        y_true_real: True labels of the binary classifier for real data.
        y_score_real: Predicted scores or probabilities of the positive class for real data.
        y_true_synth: True labels of the binary classifier for synthetic data.
        y_score_synth: Predicted scores or probabilities of the positive class for synthetic data.
        classifier_name: Name of the classifier for setting the title of the plot."""

    # Compute the false positive rate, true positive rate, and thresholds for real data
    fpr_real, tpr_real, thresholds_real = roc_curve(y_true_real, y_score_real)
    # Compute the false positive rate, true positive rate, and thresholds for synthetic data
    fpr_synth, tpr_synth, thresholds_synth = roc_curve(y_true_synth, y_score_synth)
    # Compute the area under the ROC curve for real data
    roc_auc_real = auc(fpr_real, tpr_real)
    # Compute the area under the ROC curve for synthetic data
    roc_auc_synth = auc(fpr_synth, tpr_synth)

    ax.plot(fpr_real, tpr_real, color='b', label='Real (AUC = %0.2f)' % roc_auc_real)
    ax.plot(fpr_synth, tpr_synth, color='g', label='Synthetic (AUC = %0.2f)' % roc_auc_synth)
    ax.plot([0, 1], [0, 1], color='r', linestyle='--', lw=2, label='Random', alpha=.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for {}'.format(clf_name))
    ax.legend()


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

    fig, axes = plt.subplots(len(classifiers), 3, figsize=(15, 5*len(classifiers)), squeeze=False)


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

                plot_roc_binary(y_true, y_score, y_true_2, y_score_2, f'{clf_name}_{data_name}', ax=ax)

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(results)
    return df

print(print_results(real_path, fake_path, mixed_path))
