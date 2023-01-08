"""
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2022.12.19
Last Update: 2022.12.25

[ISSUE]
1. 在 sklearn 底下 許多 metrics 下 average 的使用
    - binary: 只能用在二元資料下  
    - Micro: 直接用全部資料的 true positives, false negatives and false positives 計算
    - Macro: 直接加權平均
    - weighted: 對每個 label 乘權重的加權平均(處理 unbalance data)
    - samples: ??
"""
import functools
import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt

# pip install pretty_confusion_matrix
from pretty_confusion_matrix import pp_matrix

def plot_confusion_matrix(confusion_matrix, file_path, class_ls):
    df_cm = pd.DataFrame(confusion_matrix, index=class_ls, columns=class_ls)
    pp_matrix(df_cm, cmap='PuRd')

    plt.savefig(file_path)

def get_evaluation(pred, label, pred_prob):
    result_dt = {k:v(y_true=label, y_pred=pred) for k, v in metrics_dt.items()}
    if pred_prob != None:
        result_dt.update({k:v(y_true=label, y_score=pred_prob) for k, v in prob_metrics_dt.items()})
    report = metrics.classification_report(y_true=label, y_pred=pred, output_dict=True)

    return result_dt, report

# y_true, y_pred -> [0, 1, 1, 0], [0, 1, 1, 1]
metrics_dt = {
    'Weighted_Precision': functools.partial(metrics.precision_score, average='weighted'),
    'Balanced_acc': metrics.balanced_accuracy_score,
    'f1_micro': functools.partial(metrics.f1_score, average='micro'),
    'f1_macro': functools.partial(metrics.f1_score, average='macro'),
    'f1_weighted': functools.partial(metrics.f1_score, average='weighted')
}


# y_true, y_score -> [0, 1, 2], [[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3]]
prob_metrics_dt = {
    'Top_3_acc': functools.partial(metrics.top_k_accuracy_score, k=3),
    'Top_5_acc': functools.partial(metrics.top_k_accuracy_score, k=5),
    'Top_10_acc': functools.partial(metrics.top_k_accuracy_score, k=10),
    'roc_auc_score(ovr)': functools.partial(metrics.roc_auc_score, multi_class='ovr'),
    'roc_auc_score(ovo)': functools.partial(metrics.roc_auc_score, multi_class='ovo')
}