# metrics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, confusion_matrix
)

def best_threshold_from_pr(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    if len(f1s) <= 1 or len(thr) == 0:
        return 0.5
    best_idx = np.nanargmax(f1s[1:]) + 1  # align with thresholds
    return float(thr[best_idx - 1])

def plot_roc(y_true, y_score, label, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_pr(y_true, y_score, label, out_png):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6,4))
    plt.plot(rec, prec, label=label)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve'); plt.legend(loc='lower left'); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_cm(y_true, y_pred, label, out_png):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Confusion Matrix: {label}')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['No-Conflict','Conflict'])
    plt.yticks(ticks, ['No-Conflict','Conflict'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()
