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

    def delete(self, clf, train_data, test_data, test_size=0.25):

        x, y = self.read_data(train_data)
        self.classes = np.unique(y)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=16)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)  # Apply scaling to X_test

        clf.fit(X_train, y_train)

        # predict on the dataset it is trained on
        y_pred = clf.predict(X_test)

        y_pred_probs = clf.predict_proba(X_test)  # Predicted probabilities of all classes
        f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)

        # Binarize the true labels, in case it is a multiclass problem
        y_true = label_binarize(y_test, classes=self.classes)

        # test data
        x_val, y_val = self.read_data(test_data)

        x_val = self.scaler.transform(x_val)  # Apply scaling to x_val

        y_pred_2 = clf.predict(x_val)

        y_pred_probs_2 = clf.predict_proba(x_val)  # Predicted probabilities of all classes
        f1_2 = round(f1_score(y_val, y_pred_2, average='weighted'), 4)

        # Binarize the true labels, in case it is a multiclass problem
        y_true_2 = label_binarize(y_val, classes=self.classes)

        # f1, y_true and y_pred_probs are results from testing on the train dataset
        # f1_2, y_true_2 and y_pred_probs_2 are from testing on the test data
        return f1, f1_2, y_true, y_true_2, y_pred_probs, y_pred_probs_2

    def evaluate_classifier(self, clf, X_test, y_test):
        # predict
        y_pred = clf.predict(X_test)

        # Predicted probabilities of all classes
        y_pred_probs = clf.predict_proba(X_test)
        f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)

        # Binarize the true labels, in case it is a multiclass problem
        y_true = label_binarize(y_test, classes=self.classes)

        return f1, y_true, y_pred, y_pred_probs

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
        f1, y_true, y_pred, y_pred_probs = self.evaluate_classifier(clf, X_test, y_test)

        # results from testing on the test data
        f1_2, y_true_2, y_pred_2, y_pred_probs_2 = self.evaluate_classifier(clf, X_test_2, y_test_2)

        return f1, f1_2, y_true, y_true_2, y_pred_probs, y_pred_probs_2

    def train_test_cross_val(self, clf, train_data, test_data, n_folds=5):
        """
        Train the classifier on one dataset using cross-validation and test on both train and test datasets.

        :param clf: Classifier object
        :param train_data: Path to the dataset file for training
        :param test_data: Path to the dataset file for testing
        :param n_folds: Number of cross-validation folds
        :return: Average F1 score across folds, true labels, and predicted probabilities for train and test datasets
        """

        x_train_raw, y_train_raw = self.read_data(train_data)
        x_test_raw, y_test_raw = self.read_data(test_data)
        self.classes = np.unique(y_train_raw)

        x_train_raw = self.scaler.fit_transform(x_train_raw)
        x_test_raw = self.scaler.transform(x_test_raw)

        cv = StratifiedKFold(n_splits=n_folds)

        cv_scores_train = []
        y_pred_probs_train = []
        y_true_train = []

        cv_scores_test = []
        y_pred_probs_test = []
        y_true_test = []

        for train_idx, test_idx in cv.split(x_train_raw, y_train_raw):
            x_train, y_train = x_train_raw[train_idx], y_train_raw[train_idx]
            x_test, y_test = x_train_raw[test_idx], y_train_raw[test_idx]

            clf.fit(x_train, y_train)

            # predict and get F1 score for train dataset
            y_pred_train = clf.predict(x_train)
            f1_train = f1_score(y_train, y_pred_train, average='weighted')
            cv_scores_train.append(f1_train)

            y_pred_prob_train = clf.predict_proba(x_train)
            y_pred_probs_train.append(y_pred_prob_train)

            y_true_train_fold = label_binarize(y_train, classes=self.classes)
            y_true_train.append(y_true_train_fold)

            # predict and get F1 score for test dataset
            y_pred_test = clf.predict(x_test_raw)
            f1_test = f1_score(y_test_raw, y_pred_test, average='weighted')
            cv_scores_test.append(f1_test)

            y_pred_prob_test = clf.predict_proba(x_test_raw)
            y_pred_probs_test.append(y_pred_prob_test)

            y_true_test_fold = label_binarize(y_test_raw, classes=self.classes)
            y_true_test.append(y_true_test_fold)

        # Average the predicted probabilities for train and test datasets from each fold
        y_pred_probs_train = np.mean(y_pred_probs_train, axis=0)
        y_pred_probs_test = np.mean(y_pred_probs_test, axis=0)

        y_true_train = np.concatenate(y_true_train)
        y_true_test = np.concatenate(y_true_test)

        return np.round(np.mean(cv_scores_train), 5), np.round(np.mean(cv_scores_test), 5), \
               y_true_train, y_true_test, y_pred_probs_train, y_pred_probs_test

    def train_test_cross_val_2(self, clf, train_data, test_data, n_folds=5):
        x_train_raw, y_train_raw = self.read_data(train_data)
        self.classes = np.unique(y_train_raw)

        x_train_raw = self.scaler.fit_transform(x_train_raw)

        cv = StratifiedKFold(n_splits=n_folds)

        cv_scores = []
        y_pred_probs = []

        for train_idx, test_idx in cv.split(x_train_raw, y_train_raw):
            x_train, y_train = x_train_raw[train_idx], y_train_raw[train_idx]
            X_test, y_test = x_train_raw[test_idx], y_train_raw[test_idx]

            clf.fit(x_train, y_train)

            y_pred = clf.predict(X_test)
            f1_2 = f1_score(y_test, y_pred, average='weighted')
            cv_scores.append(f1_2)

            y_pred_prob = clf.predict_proba(X_test)
            y_pred_probs.append(y_pred_prob)

        # Average the predicted probabilities for dataset2 from each fold
        y_pred_probs = np.mean(y_pred_probs, axis=0)

        # Binarize the true labels for dataset2
        y_true = label_binarize(y_train_raw, classes=self.classes)

        return np.round(np.mean(cv_scores), 5), y_true, y_pred_probs

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
