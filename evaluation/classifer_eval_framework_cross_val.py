import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities.plots import plot_multiclass_roc, plot_binaryclass_roc

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import label_binarize, MinMaxScaler

"""
This framework provides an evaluation platform for comparing the 
performance of different classifiers on real, synthetic, and mixed datasets. 
It enables users to add classifiers, preprocess data, and conduct train-test 
evaluations using various combinations of datasets. The results, including F1 scores and ROC curves, 
are visualized and saved for easy comparison and analysis.
"""

class ClassifierEvaluationFrameworkCrossVal:

    def __init__(self):
        self.classes = None
        self.scaler = MinMaxScaler()
        self.classifiers = []

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

    def train_on_1_test_on_2(self, clf, dataset1, dataset2, n_splits=5):
        """
        Train the classifier on one dataset using cross-validation and test on another dataset.

        :param clf: Classifier object
        :param dataset1: Path to the first dataset file (for training)
        :param dataset2: Path to the second dataset file (for testing)
        :param n_splits: Number of cross-validation folds
        :return: Average F1 score across folds, true labels, and predicted probabilities
        """

        x, y = self.read_data(dataset1)
        self.classes = np.unique(y)

        x = self.scaler.fit_transform(x)

        x_test, y_test = self.read_data(dataset2)
        x_test = self.scaler.transform(x_test)

        # Split dataset2 into training and validation sets
        x_val, x_train2, y_val, y_train2 = train_test_split(x_test, y_test, test_size=0.2, stratify=y_test,
                                                            random_state=42)

        cv = StratifiedKFold(n_splits=n_splits)

        cv_scores = []
        y_pred_probs = []

        for train_idx, _ in cv.split(x, y):
            x_train, y_train = x[train_idx], y[train_idx]

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_val)
            f1_2 = f1_score(y_val, y_pred, average='weighted')
            cv_scores.append(f1_2)

            y_pred_prob = clf.predict_proba(x_val)
            y_pred_probs.append(y_pred_prob)

        # Average the predicted probabilities for dataset2 from each fold
        y_pred_probs = np.mean(y_pred_probs, axis=0)

        # Binarize the true labels for dataset2
        y_true = label_binarize(y_val, classes=self.classes)

        return np.round(np.mean(cv_scores), 5), y_true, y_pred_probs


    def t1t2_results(self, real_data, synth_data, mixed_data, result_path):
        """
        Evaluate classifiers using the train_on_1_test_on_2() function with all combinations of datasets and classifiers.

        :param real_data: Path to the real dataset file
        :param synth_data: Path to the synthetic dataset file
        :param mixed_data: Path to the mixed dataset file
        :param result_path: Path to the directory for storing results
        :param test_size: Proportion of the dataset to include in the test split
        """

        if not self.classifiers:
            raise ValueError("Please provide at least one classifier")

        results = []

        # Split the classifiers into groups of 3
        classifier_groups = [self.classifiers[i:i + 3] for i in range(0, len(self.classifiers), 3)]

        for group_idx, classifier_group in enumerate(classifier_groups):
            fig, axes = plt.subplots(len(classifier_group), 3, figsize=(15, 5 * len(classifier_group)), squeeze=False)

            for idx, (clf_name, clf) in enumerate(classifier_group):

                data_dict = {
                    "real": (real_data, real_data, synth_data),
                    "synth": (synth_data, real_data, synth_data),
                    "mixed": (mixed_data, real_data, mixed_data)
                }

                for jdx, (data_name, (train_data, test_data1, test_data2)) in enumerate(data_dict.items()):

                    print(f"Training classifier {clf_name} on {data_name} data")

                    ax = axes[idx, jdx]

                    f1, y_true, y_score = self.train_on_1_test_on_2(clf, train_data, test_data1)
                    f1_2, y_true_2, y_score_2 = self.train_on_1_test_on_2(clf, train_data,
                                                                          test_data2)

                    results.append({
                        'classifier': clf_name,
                        'train_data': data_name,
                        'f1_real': f1,
                        'f1_synth': f1_2
                    })

                    n_classes = len(self.classes)

                    y_true = label_binarize(y_true, classes=self.classes)
                    y_true_2 = label_binarize(y_true_2, classes=self.classes)

                    if n_classes == 2:
                        y_score = y_score[:, 1]
                        y_score_2 = y_score_2[:, 1]

                        plot_binaryclass_roc(y_true, y_score, y_true_2, y_score_2, n_classes, f'{clf_name}_{data_name}',
                                             ax=ax)

                    else:
                        plot_multiclass_roc(y_true, y_score, y_true_2, y_score_2, n_classes, f'{clf_name}_{data_name}',
                                            ax=ax, plot_class_curves=False)

            plt.tight_layout()
            plt.savefig(f'{result_path}/roc_curves_cross_val_{group_idx + 1}.png')
            plt.show()

        df = pd.DataFrame(results)
        df.to_csv(f'{result_path}/classifier_results_cross_val.csv', index=False)
