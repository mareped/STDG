import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_roc_binaryclass(y_true_real, y_score_real, y_true_synth, y_score_synth, clf_name, ax=None):
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


def plot_multiclass_roc(y_true_1, y_score_1, n_classes, plot_class_curves=True):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_1[:, i], y_score_1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_1.ravel(), y_score_1.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (AUC = {0:0.2f})'
                                               ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    if plot_class_curves:
        colors = plt.cm.get_cmap('tab10', n_classes)(range(n_classes)).tolist()
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    if plot_class_curves:
        plt.legend(loc="lower right")
    else:
        plt.legend(loc="best")
    plt.show()
