import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities.plots import plot_multiclass_roc, plot_binaryclass_roc
from config import DataConfig

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize


class ClassifierEvaluator:
    def __init__(self):
        self.classes = None

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

    def evaluate_classifier(self, clf, dataset1, dataset2):
        X_train, X_test, y_train, y_test = self.split_data(dataset1)
        x_2, y_2 = self.read_data(dataset2)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(x_2)

        if len(self.classes) == 2:
            y_pred_probs = clf.predict_proba(x_2)[:, 1]  # Predicted probabilities of the positive class
        else:
            y_pred_probs = clf.predict_proba(x_2)  # Predicted probabilities of all classes

        # Binarize the true labels, in case it is a multiclass problem
        y_true = label_binarize(y_2, classes=self.classes)

        f1 = f1_score(y_2, y_pred, average='weighted')

        return f1, y_true, y_pred_probs

    def print_results(self, real_data, synth_data, mixed_data):

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
            for jdx, (data_name, data) in enumerate(
                    {"real": real_data, "synth": synth_data, "mixed": mixed_data}.items()):

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
                    f1, y_true, y_score = self.evaluate_classifier(clf, train_data, test_data1)
                    f1_2, y_true_2, y_score_2 = self.evaluate_classifier(clf, train_data, test_data2)

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
        plt.show()

        df = pd.DataFrame(results)
        return df


# create a new instance of DataConfig, with the dataset (lower_back_pain, obesity)
config = DataConfig(dataset_name='obesity', model_name='ctgan', epochs=800, batch_size=100)

real_path, fake_path, mixed_path = config.real_path, config.fake_path, config.mixed_path

evaluator = ClassifierEvaluator()
evaluator.print_results(real_path, fake_path, mixed_path)
