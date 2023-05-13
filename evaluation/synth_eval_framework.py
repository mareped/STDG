import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities.plots import plot_multiclass_roc, plot_binaryclass_roc

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler


"""
This framework provides an evaluation platform for comparing the 
performance of different classifiers on real, synthetic, and mixed datasets. 
It enables users to add classifiers, preprocess data, and conduct train-test or cross validation
evaluations using various combinations of datasets. The results, including F1 scores and ROC curves, 
are visualized and saved for easy comparison and analysis.
"""


class SynthEval:

    def __init__(self, real_path, synth_path, result_path):
        self.real_path = real_path
        self.synth_path = synth_path
        self.result_path = result_path + "/classifier_evaluation"
        self.augmented_data = None
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

    def get_features_and_labels(self, dataset):
        """
        Read dataset from the specified path and return features and labels.

        :param dataset_path: Path to the dataset file
        :return: Features and labels as NumPy arrays
        """

        # in case of augmented data
        if dataset is None:
            return None, None

        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset)

        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        return x, y

    def create_augmented_data(self, real_percentage, synth_percentage, random_seed=None):
        """
        Create an augmented dataset by combining a percentage of the real data and a percentage of the fake data.

        Args:
            real_percentage (float): The percentage of the real data to use for augmentation (between 0 and 1).
            synth_percentage (float): The percentage of the fake data to use for augmentation (between 0 and 1).
            random_seed (int, optional): Seed for reproducibility.

        Returns:
            tuple: A tuple containing the remaining real data and the augmented data.
        """

        real_data = pd.read_csv(self.real_path)
        synth_data = pd.read_csv(self.synth_path)
        if random_seed is not None:
            np.random.seed(random_seed)

        real_data = real_data.sample(frac=1).reset_index(drop=True)

        real_sample_size = int(len(real_data) * real_percentage)
        synth_sample_size = int(len(synth_data) * synth_percentage)

        real_sample = real_data.sample(n=real_sample_size)
        remaining_real_data = real_data.drop(real_sample.index)

        synth_sample = synth_data.sample(n=synth_sample_size)

        self.augmented_data = pd.concat([real_sample, synth_sample]).sample(frac=1).reset_index(drop=True)

        return remaining_real_data, self.augmented_data

    def train_test(self, clf, train_data, test_data, test_size=0.25):
        x_train_raw, y_train_raw = self.get_features_and_labels(train_data)
        self.classes = np.unique(y_train_raw)

        # defines test data
        X_test_2, y_test_2 = self.get_features_and_labels(test_data)

        # in case of augmented data
        if x_train_raw is None and y_train_raw is None:
            x_train_raw, y_train_raw = self.get_features_and_labels(self.augmented_data)

        X_train, X_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=test_size,
                                                            random_state=16)

        # min_max data
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_test_2 = self.scaler.transform(X_test_2)

        clf.fit(X_train, y_train)

        # test on the train_data
        y_pred = clf.predict(X_test)
        f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)
        y_pred_probs = clf.predict_proba(X_test)

        # test on the test_Data
        y_pred_2 = clf.predict(X_test_2)
        f1_2 = round(f1_score(y_test_2, y_pred_2, average='weighted'), 4)
        y_pred_probs_2 = clf.predict_proba(X_test_2)

        return f1, f1_2, y_test, y_test_2, y_pred_probs, y_pred_probs_2

    def train_test_cross_val(self, clf, train_data, test_data, n_folds=5):
        """
        Train and test the classifier using k-fold cross-validation.

        :param clf: Classifier object to train and test
        :param train_data: Path to the training dataset file
        :param test_data: Path to the testing dataset file
        :param n_folds: Number of folds for cross-validation
        :return: f1 score for the training data and testing data, true labels for both datasets,
                 predicted probabilities for both datasets
        """

        # Read the training dataset and scale the features
        x_train_raw, y_train_raw = self.get_features_and_labels(train_data)
        self.classes = np.unique(y_train_raw)

        # in case of augmented data
        if x_train_raw is None and y_train_raw is None:
            x_train_raw, y_train_raw = self.get_features_and_labels(self.augmented_data)

        # Read the testing dataset
        X_test_2, y_test_2 = self.get_features_and_labels(test_data)

        # min_max data
        x_train_raw = self.scaler.fit_transform(x_train_raw)
        X_test_2 = self.scaler.transform(X_test_2)

        kf = StratifiedKFold(n_splits=n_folds)

        y_preds_all = []
        y_pred_probs_all = []
        y_pred_probs_all_2 = []

        for train_index, test_index in kf.split(x_train_raw, y_train_raw):
            # Split the data into training and testing sets
            X_train, X_test = x_train_raw[train_index], x_train_raw[test_index]
            y_train, y_test = y_train_raw[train_index], y_train_raw[test_index]

            # Fit the classifier to the training data
            clf.fit(X_train, y_train)

            # Make predictions on the test data to the "train_data"
            y_pred = clf.predict(X_test)
            y_pred_probs = clf.predict_proba(X_test)

            # Make predictions on the testing dataset for "test_data"
            y_pred_probs_2 = clf.predict_proba(X_test_2)

            # Append the predictions and probabilities to the lists
            y_preds_all.extend(list(zip(test_index, y_pred)))
            y_pred_probs_all.extend(list(zip(test_index, y_pred_probs)))
            y_pred_probs_all_2.append(y_pred_probs_2)

        # Sort the predictions and probabilities by index to match the original order of the samples
        y_preds_all.sort(key=lambda x: x[0])
        y_pred_probs_all.sort(key=lambda x: x[0])

        # Convert the predictions and probabilities of "train_data" to NumPy arrays
        y_true = np.array(y_train_raw)
        y_pred = np.array([pred for _, pred in y_preds_all])
        y_pred_probs = np.array([probs for _, probs in y_pred_probs_all])

        # Average the probabilities for the "test_data" and compute the predictions from the averaged probabilities
        y_pred_probs_2 = np.mean(y_pred_probs_all_2, axis=0)
        y_preds_2 = np.argmax(y_pred_probs_2, axis=1)

        # Compute the f1 score for both datasets
        f1 = round(f1_score(y_true, y_pred, average='weighted'), 4)
        f1_2 = round(f1_score(y_test_2, y_preds_2, average='weighted'), 4)

        return f1, f1_2, y_true, y_test_2, y_pred_probs, y_pred_probs_2

    def compare_datasets_performance(self, real_percentage=0.5, synth_percentage=1, cross_val=False):
        """
        Evaluate classifiers using either the train_test() and train_test_cross_val() function with all combinations of datasets and
        classifiers.

        :param real_percentage: How much of the real data to use for augmentation
        :param synth_percentage: How much of the synthetic data to use for augmentation
        :param cross_val: Uses cross-validation instead of Train-Test split if True
        """

        if not self.classifiers:
            raise ValueError("Please provide at least one classifier")

        # Split the classifiers into groups of 3 (for plotting)
        classifier_groups = [self.classifiers[i:i + 3] for i in range(0, len(self.classifiers), 3)]

        results = []

        for group_idx, classifier_group in enumerate(classifier_groups):
            fig, axes = plt.subplots(len(classifier_group), 3, figsize=(15, 5 * len(classifier_group)), squeeze=False)

            for idx, (clf_name, clf) in enumerate(classifier_group):

                data_dict = {
                    "real": (self.real_path, self.synth_path),
                    "synth": (self.synth_path, self.real_path),
                    "augmented": (None, self.real_path)  # Set train_data to None for augmented case
                }

                for jdx, (data_name, (train_data, test_data)) in enumerate(data_dict.items()):
                    ax = axes[idx, jdx]

                    print(f"Training classifier {clf_name} on {data_name} data")

                    real_data_case = False  # Add this line to create a flag for the real data case

                    if data_name == "augmented":
                        # Create the augmented dataset
                        remaining_real_data, train_data = self.create_augmented_data(real_percentage, synth_percentage)
                        test_data = remaining_real_data
                    else:
                        real_data_case = train_data == self.real_path  # Set the flag if train_data is the real_path

                    if cross_val:
                        f1, f1_2, y_true, y_true_2, y_score, y_score_2 = self.train_test_cross_val(clf, train_data,
                                                                                                   test_data)

                    else:
                        f1, f1_2, y_true, y_true_2, y_score, y_score_2 = self.train_test(clf, train_data, test_data)

                    # if train data == real path, then f1_real = f1 and f1_synth_or_augmented = f1_2. And opposite.
                    f1_real, f1_synth_or_augmented = (f1, f1_2) if real_data_case else (f1_2, f1)

                    f1_difference = round(f1_real - f1_synth_or_augmented, 4)

                    results.append({
                        'classifier': clf_name,
                        'train_data': data_name,
                        'f1_real': f1_real,
                        'f1_synth/augmented': f1_synth_or_augmented,
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
            plt.savefig(f'{self.result_path}/roc_curves_cross_val_{group_idx + 1}.png')
            plt.show()

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{self.result_path}/classifier_f1_scores_cross_val.csv', index=False)
