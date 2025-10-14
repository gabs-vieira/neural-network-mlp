import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_metrics(y_true, y_pred):
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return pd.DataFrame([{'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'accuracy': accuracy,
                          'precision': precision, 'recall': recall, 'f1': f1}])

def plot_loss(losses, path='training_loss.png'):
    plt.figure()
    plt.plot(losses)
    plt.title('Loss por época')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_roc(y_true, y_scores, path='roc_curve.png'):
    thresholds = np.linspace(0, 1, 101)
    tprs, fprs = [], []
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        TP = ((y_true == 1) & (preds == 1)).sum()
        TN = ((y_true == 0) & (preds == 0)).sum()
        FP = ((y_true == 0) & (preds == 1)).sum()
        FN = ((y_true == 1) & (preds == 0)).sum()
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    auc = np.trapz(tprs, fprs)
    plt.figure()
    plt.plot(fprs, tprs, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
