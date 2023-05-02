import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities.plots import plot_multiclass_roc, plot_binaryclass_roc

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize, MinMaxScaler, LabelBinarizer

import warnings

"""
This framework provides an evaluation platform for comparing the 
performance of different classifiers on real, synthetic, and mixed datasets. 
It enables users to add classifiers, preprocess data, and conduct train-test or cross validation
evaluations using various combinations of datasets. The results, including F1 scores and ROC curves, 
are visualized and saved for easy comparison and analysis.
"""


class ClassifierEvaluationFramework:

    def __init__(self, real_path, synth_path, mixed_path, result_path):
        self.real_path = real_path
        self.synth_path = synth_path
        self.mixed_path = mixed_path
        self.result_path = result_path + "/classifier_evaluation"
        self.classes = None
        self.scaler = MinMaxScaler()
        self.classifiers = []

        # Create the directory if it doesn't exist
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def add_classifier(self, clf):
        """
        Add a classifier to the list of classifiers to train.

        :param clf: Classifier object
        """

        self.classifiers.append((clf.__class__.__name__, clf))

    def add_all_classifiers(self, *classifiers):
        """
        Add all classifiers at once to use for training.

        :param classifiers: Classifier objects
        """
        for clf in classifiers:
            self.add_classifier(clf)

    def read_data(self, dataset_path):
        """
        Read dataset from the specified path and return features and labels.

        :param dataset_path: Path to the dataset file
        :return: Features and labels as NumPy arrays
        """

        dataset = pd.read_csv(dataset_path)

        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        return x, y

    def evaluate_classifier(self, clf, X_test, y_test):
        # predict
        y_pred = clf.predict(X_test)

        # Predicted probabilities of all classes
        y_pred_probs = clf.predict_proba(X_test)
        f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)

        return f1, y_pred, y_pred_probs

    def train_test(self, clf, train_data, test_data, test_size=0.25):
        x_train_raw, y_train_raw = self.read_data(train_data)
        self.classes = np.unique(y_train_raw)

        X_train, X_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=test_size,
                                                            random_state=16)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # test data
        X_test_2, y_test_2 = self.read_data(test_data)
        X_test_2 = self.scaler.transform(X_test_2)

        clf.fit(X_train, y_train)

        # results from testing on the same data as the train set
        f1, y_pred, y_pred_probs = self.evaluate_classifier(clf, X_test, y_test)

        # results from testing on the test data
        f1_2, y_pred_2, y_pred_probs_2 = self.evaluate_classifier(clf, X_test_2, y_test_2)

        return f1, f1_2, y_test, y_test_2, y_pred_probs, y_pred_probs_2

    def train_test_cross_val(self, clf, train_data, test_data, n_folds=5):
        x_raw, y_raw = self.read_data(train_data)
        X = self.scaler.fit_transform(x_raw)
        self.classes = np.unique(y_raw)  # Set self.classes to the unique class labels

        # test data
        X_test_2, y_test_2 = self.read_data(test_data)
        X_test_2 = self.scaler.transform(X_test_2)

        kf = StratifiedKFold(n_splits=n_folds)

        y_preds_all = []
        y_pred_probs_all = []
        y_pred_probs_all_2 = []

        for train_index, test_index in kf.split(X, y_raw):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_raw[train_index], y_raw[test_index]

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            y_pred_probs = clf.predict_proba(X_test)
            y_pred_probs_2 = clf.predict_proba(X_test_2)

            y_preds_all.append(dict(zip(test_index, y_pred)))
            y_pred_probs_all.append(dict(zip(test_index, y_pred_probs)))
            y_pred_probs_all_2.append(y_pred_probs_2)

        # Combine the predictions and probabilities from all folds
        y_preds_combined = {index: pred for fold_preds in y_preds_all for index, pred in fold_preds.items()}
        y_pred_probs_combined = {index: probs for fold_probs in y_pred_probs_all for index, probs in fold_probs.items()}

        # Sort the predictions and probabilities by index to match the original order of the samples
        y_true = np.array(y_raw)
        y_pred = np.array([y_preds_combined[i] for i in sorted(y_preds_combined)])
        y_pred_probs = np.array([y_pred_probs_combined[i] for i in sorted(y_pred_probs_combined)])

        # Average the probabilities for dataset2 and compute the predictions from the averaged probabilities
        y_pred_probs_2_avg = np.mean(y_pred_probs_all_2, axis=0)
        y_preds_2_avg = np.argmax(y_pred_probs_2_avg, axis=1)

        f1 = f1_score(y_true, y_pred, average='weighted')
        f1_2 = f1_score(y_test_2, y_preds_2_avg, average='weighted')

        return f1, f1_2, y_true, y_test_2, y_pred_probs, y_pred_probs_2_avg


    def t1t2_results(self, cross_val=False):
        """
        Evaluate classifiers using the train_on_1_test_on_2() function with all combinations of datasets and
        classifiers.

        :param cross_val: Uses cross-validation instead of Train-Test split if True
        :param real_path: Path to the real dataset file
        :param synth_path: Path to the synthetic dataset file
        :param mixed_path: Path to the mixed dataset file
        :param result_path: Path to the directory for storing results
        :param test_size: Proportion of the dataset to include in the test split
        """

        warnings.filterwarnings("ignore")

        if not self.classifiers:
            raise ValueError("Please provide at least one classifier")

        # Split the classifiers into groups of 3
        classifier_groups = [self.classifiers[i:i + 3] for i in range(0, len(self.classifiers), 3)]

        results = []

        for group_idx, classifier_group in enumerate(classifier_groups):
            fig, axes = plt.subplots(len(classifier_group), 3, figsize=(15, 5 * len(classifier_group)), squeeze=False)

            for idx, (clf_name, clf) in enumerate(classifier_group):

                data_dict = {
                    "real": (self.real_path, self.synth_path),
                    "synth": (self.synth_path, self.real_path),
                    "mixed": (self.mixed_path, self.real_path)
                }

                for jdx, (data_name, (train_data, test_data)) in enumerate(data_dict.items()):
                    ax = axes[idx, jdx]

                    print(f"Training classifier {clf_name} on {data_name} data")

                    if cross_val:
                        f1, f1_2, y_true, y_true_2, y_score, y_score_2 = self.train_test_cross_val(clf, train_data,
                                                                                                      test_data)

                    else:
                        f1, f1_2, y_true, y_true_2, y_score, y_score_2 = self.train_test(clf, train_data, test_data)

                    # if train data == real path, then f1_real = f1 and f1_synth_or_mixed = f1_2. And opposite.
                    f1_real, f1_synth_or_mixed = (f1, f1_2) if train_data == self.real_path else (f1_2, f1)

                    f1_difference = f1_real - f1_synth_or_mixed

                    results.append({
                        'classifier': clf_name,
                        'train_data': data_name,
                        'f1_real': f1_real,
                        'f1_synth/mixed': f1_synth_or_mixed,
                        'f1_difference': f1_difference
                    })

                    n_classes = len(self.classes)

                    if n_classes == 2:
                        y_score = y_score[:, 1]
                        y_score_2 = y_score_2[:, 1]
                        plot_binaryclass_roc(y_true, y_score, y_true_2, y_score_2, n_classes, f'{clf_name}_{data_name}',
                                             ax=ax)

                    else:
                        plot_multiclass_roc(y_true, y_score, y_true_2, y_score_2, n_classes, f'{clf_name}_{data_name}',
                                            ax=ax, plot_class_curves=False)

            plt.tight_layout()
            # plt.savefig(f'{self.result_path}/roc_curves_x_val{group_idx + 1}.png')
            plt.show()

        results_df = pd.DataFrame(results)  # Convert the list of dictionaries to a DataFrame
        print(results_df)
