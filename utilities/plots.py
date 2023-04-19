import numpy as np
from sklearn.metrics import auc, roc_curve


def compute_roc_auc(y_true, y_score, n_classes):
    # Multiclass problem
    if n_classes > 2:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Binary class problem
    else:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def plot_binaryclass_roc(y_true_real, y_score_real, y_true_synth, y_score_synth, n_classes, clf_name, ax=None):
    """Args:
        y_true_real: True labels of the binary classifier for real data.
        y_score_real: Predicted scores or probabilities of the positive class for real data.
        y_true_synth: True labels of the binary classifier for synthetic data.
        y_score_synth: Predicted scores or probabilities of the positive class for synthetic data.
        clf_name: Name of the classifier for setting the title of the plot.
        n_classes: number of classes for the classification problem"""

    fpr_real, tpr_real, roc_auc_real = compute_roc_auc(y_true_real, y_score_real, n_classes)
    fpr_synth, tpr_synth, roc_auc_synth = compute_roc_auc(y_true_synth, y_score_synth, n_classes)

    ax.plot(fpr_real, tpr_real, color='b', label='Real (AUC = %0.2f)' % roc_auc_real)
    ax.plot(fpr_synth, tpr_synth, color='g', label='Synthetic (AUC = %0.2f)' % roc_auc_synth)
    ax.plot([0, 1], [0, 1], color='r', linestyle='--', lw=2, label='Random', alpha=.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for {}'.format(clf_name))
    ax.legend()


def plot_multiclass_roc(y_true_real, y_score_real, y_true_synth, y_score_synth, n_classes, clf_name,
                        plot_class_curves=True, ax=None):
    fpr_real, tpr_real, roc_auc_real = compute_roc_auc(y_true_real, y_score_real, n_classes)
    fpr_synth, tpr_synth, roc_auc_synth = compute_roc_auc(y_true_synth, y_score_synth, n_classes)
    lw = 2
    ax.plot(fpr_real["micro"], tpr_real["micro"], label='micro-average - Real (AUC = {0:0.2f})'
                                                        ''.format(roc_auc_real["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    ax.plot(fpr_real["macro"], tpr_real["macro"],
            label='macro-average - Real (AUC = {0:0.2f})'.format(roc_auc_real["macro"]),
            color='navy', linestyle=':', linewidth=4)
    ax.plot(fpr_synth["macro"], tpr_synth["macro"],
            label='macro-average - Synthetic (AUC = {0:0.2f})'.format(roc_auc_synth["macro"]),
            color='darkorange', linestyle=':', linewidth=4)
    if plot_class_curves:
        colors_real = ax.cm.get_cmap('tab10', n_classes)(range(n_classes)).tolist()
        colors_synth = ax.cm.get_cmap('tab10', n_classes)(range(n_classes)).tolist()
        [ax.plot(fpr_real[i], tpr_real[i], color=colors_real[i], lw=lw) for i in range(n_classes)]
        [ax.plot(fpr_synth[i], tpr_synth[i], color=colors_synth[i], lw=lw, linestyle='--',
                 label='Class {0} - Synthetic (AUC = {1:0.2f})'.format(i, roc_auc_synth[i])) for i in range(n_classes)]

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for {}'.format(clf_name))
    ax.legend(loc="lower right")