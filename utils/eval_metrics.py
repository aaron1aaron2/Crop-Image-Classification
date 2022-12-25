"""
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2022.12.19
Last Update: 2022.12.25
"""
import functools
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from pretty_confusion_matrix import pp_matrix

# y_true, y_score -> [0, 1, 2], [[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3]]
prob_metrics_dt = {
    'Top_3_acc': functools.partial(metrics.top_k_accuracy_score, k=3), # y_true, y_pred(prob)
    'Top_5_acc': functools.partial(metrics.top_k_accuracy_score, k=5),
    'Top_10_acc': functools.partial(metrics.top_k_accuracy_score, k=10),
    'roc_auc_score(ovr)': functools.partial(metrics.roc_auc_score, multi_class='ovr'),
    'roc_auc_score(ovo)': functools.partial(metrics.roc_auc_score, multi_class='ovo')
}

# y_true, y_pred -> [0, 1, 1, 0], [0, 1, 1, 1]
metrics_dt = {
    'Balanced_acc': metrics.balanced_accuracy_score, # y_true, y_pred)
    'f1_micro': functools.partial(metrics.f1_score, average='micro'),
    'f1_macro': functools.partial(metrics.f1_score, average='macro'),
    'f1_weighted': functools.partial(metrics.f1_score, average='weighted'),
    'report': functools.partial(metrics.classification_report, output_dict=True),
    'confusion_matrix': metrics.confusion_matrix
}


def plot_confusion_matrix(confusion_matrix, file_path, class_ls):
    df_cm = pd.DataFrame(confusion_matrix, index=class_ls, columns=class_ls)
    pp_matrix(df_cm, cmap='PuRd')

    plt.savefig(file_path)

def Weighted_Precision():
    metrics.confusion_matrix