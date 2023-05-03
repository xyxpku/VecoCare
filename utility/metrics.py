from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,auc, roc_curve, precision_recall_curve
import numpy as np
from scipy import interp

def get_metrics(predict_all,targets_all, flag, epoch_id, result_path):
    result = dict()
    result["pr"] = get_metrics_pr(predict_all, targets_all)
    result["auc"] = get_metric_auc(predict_all, targets_all)
    with open(result_path, "a") as f:
        if flag == "train":
            f.write("train_epoch{}:".format(epoch_id) + str(result) + '\n')
        elif flag == "val":
            f.write("val_epoch{}:".format(epoch_id) + str(result) + '\n')
        elif flag == "test":
            f.write("test_epoch{}:".format(epoch_id)+str(result)+'\n')

    return result["pr"],result["auc"]

def get_metrics_pr(predict_all, targets_all):
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(predict_all.shape[1]):

        precision[i], recall[i], _ = precision_recall_curve(
            targets_all[:, i], predict_all[:, i])
        precision[i][np.isnan(precision[i])] = 0
        recall[i][np.isnan(recall[i])] = 0
        pr_auc[i] = auc(recall[i], precision[i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        targets_all.ravel(), predict_all.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    macro = 0.0
    for i in range(predict_all.shape[1]):
        macro += pr_auc[i]
    pr_auc['macro'] = macro / predict_all.shape[1]

    return pr_auc


def get_metric_auc(predict_all,targets_all):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(predict_all.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(
            targets_all[:, i], predict_all[:, i])
        fpr[i][np.isnan(fpr[i])] = 0
        tpr[i][np.isnan(tpr[i])] = 0
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(
        targets_all.ravel(), predict_all.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate(
        [fpr[i] for i in range(predict_all.shape[1])]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(predict_all.shape[1]):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= predict_all.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc