import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities.plots import plot_multiclass_roc, plot_binaryclass_roc, plot_cmf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize, MinMaxScaler


class ClassifierFramework:
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=42),
        "Logistic Regression": LogisticRegression(C=100, penalty='l2', solver='newton-cg'),
        "MLP Classifier": MLPClassifier(max_iter=300, activation='relu', solver='adam')
    }

    def __init__(self):
        self.classes = None
        self.scaler = MinMaxScaler()

    def read_data(self, dataset_path):
        dataset = pd.read_csv(dataset_path)

        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        return x, y

    def split_data(self, dataset_path):
        x, y = self.read_data(dataset_path)

        self.classes = np.unique(y)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

        return X_train, X_test, y_train, y_test

    def train_test(self, clf, dataset):
        X_train, X_test, y_train, y_test = self.split_data(dataset)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        clf.fit(X_train, y_train)
        return clf, X_test, y_test

    def plot_confusion_matrix(self, clf, dataset):
        trained_clf, X_test, y_test = self.train_test(clf, dataset)
        y_pred = trained_clf.predict(X_test)
        plot_cmf(y_test, y_pred, self.classes)

    def train_on_1_test_on_2(self, clf, dataset1, dataset2):
        trained_clf, _, _ = self.train_test(clf, dataset1)
        x_2, y_2 = self.read_data(dataset2)

        x_2 = self.scaler.fit_transform(x_2)

        y_pred = trained_clf.predict(x_2)

        if len(self.classes) == 2:
            y_pred_probs = clf.predict_proba(x_2)[:, 1]  # Predicted probabilities of the positive class
        else:
            y_pred_probs = clf.predict_proba(x_2)  # Predicted probabilities of all classes

        # Binarize the true labels, in case it is a multiclass problem
        y_true = label_binarize(y_2, classes=self.classes)

        f1 = f1_score(y_2, y_pred, average='weighted')

        return f1, y_true, y_pred_probs

    def print_t1t2_results(self, real_data, synth_data, mixed_data, result_path):

        classifiers = [
            ("Random Forest", RandomForestClassifier(n_estimators=1000, random_state=42)),
            ("Logistic Regression", LogisticRegression(C=100, penalty='l2', solver='newton-cg')),
            ("MLP Classifier", MLPClassifier(max_iter=300, activation='relu', solver='adam'))
        ]

        results = []
        fig, axes = plt.subplots(len(classifiers), 3, figsize=(15, 5 * len(classifiers)), squeeze=False)

        for idx, (clf_name, clf) in enumerate(classifiers):
            for jdx, (data_name, data) in enumerate(
                    {"real": real_data, "synth": synth_data, "mixed": mixed_data}.items()):

                ax = axes[idx, jdx]
                data_dict = {"real": (real_data, real_data), "synth": (synth_data, real_data),
                             "mixed": (mixed_data, real_data)}

                # train data: data it is trained on, test_data1: the first data it is tested on
                train_data, test_data1 = data_dict[data_name]
                # test_data2: second data it is tested on
                test_data2 = synth_data if data_name == "synth" else mixed_data

                f1, y_true, y_score = self.train_on_1_test_on_2(clf, train_data, test_data1)
                f1_2, y_true_2, y_score_2 = self.train_on_1_test_on_2(clf, train_data, test_data2)

                results.append({
                    'classifier': clf_name,
                    'train_data': data_name,
                    'f1_real': f1,
                    'f1_synth': f1_2
                })

                n_classes = len(self.classes)

                if n_classes == 2:
                    plot_binaryclass_roc(y_true, y_score, y_true_2, y_score_2, n_classes, f'{clf_name}_{data_name}',
                                         ax=ax)
                else:
                    plot_multiclass_roc(y_true, y_score, y_true_2, y_score_2, n_classes, f'{clf_name}_{data_name}',
                                        ax=ax, plot_class_curves=False)

        plt.tight_layout()
        plt.savefig(f'{result_path}/roc_curves.png')
        plt.show()

        df = pd.DataFrame(results)
        df.to_csv(f'{result_path}/classifier_results.csv', index=False)
