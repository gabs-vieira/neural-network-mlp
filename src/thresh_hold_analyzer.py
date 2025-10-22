# threshold_analyzer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ThresholdAnalyzer:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = y_true.flatten()
        self.y_pred_proba = y_pred_proba.flatten()

    def calculate_metrics_at_thresholds(self, thresholds=None):
        """Calcula todas as métricas para diferentes thresholds"""
        if thresholds is None:
            thresholds = np.arange(0.0, 1.05, 0.05)

        results = []

        for threshold in thresholds:
            y_pred = (self.y_pred_proba >= threshold).astype(int)

            TP = np.sum((y_pred == 1) & (self.y_true == 1))
            FP = np.sum((y_pred == 1) & (self.y_true == 0))
            TN = np.sum((y_pred == 0) & (self.y_true == 0))
            FN = np.sum((y_pred == 0) & (self.y_true == 1))

            # Métricas
            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

            results.append({
                'threshold': threshold,
                'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'fpr': fpr
            })

        return pd.DataFrame(results)

    def plot_metrics_vs_threshold(self, metrics_df, save_path=None):
        """Plot de métricas vs threshold"""
        plt.figure(figsize=(12, 8))

        plt.plot(metrics_df['threshold'], metrics_df['accuracy'], 'b-', label='Acurácia', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['precision'], 'g-', label='Precisão', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['recall'], 'r-', label='Recall', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['f1_score'], 'purple', label='F1-Score', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['specificity'], 'orange', label='Especificidade', linewidth=2)

        # Encontrar melhor threshold por F1-Score
        best_f1_idx = metrics_df['f1_score'].idxmax()
        best_threshold = metrics_df.loc[best_f1_idx, 'threshold']
        best_f1 = metrics_df.loc[best_f1_idx, 'f1_score']

        plt.axvline(x=best_threshold, color='red', linestyle='--',
                   label=f'Melhor F1 (Threshold={best_threshold:.2f})')

        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Métrica', fontsize=12)
        plt.title('Métricas vs Threshold de Classificação', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return best_threshold, best_f1