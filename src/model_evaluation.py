# evaluation_corrected.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self, y_true, y_pred_proba, thresholds=None):
        """
        y_true: valores reais (0 ou 1)
        y_pred_proba: probabilidades preditas [0,1]
        """
        self.y_true = y_true.flatten()
        self.y_pred_proba = y_pred_proba.flatten()

        if thresholds is None:
            # Gerar thresholds de 1.0 a 0.0 (importante: ordem decrescente)
            self.thresholds = np.sort(np.unique(self.y_pred_proba))[::-1]
            self.thresholds = np.append(self.thresholds, 0.0)
            self.thresholds = np.append(1.0, self.thresholds)
            self.thresholds = np.unique(self.thresholds)
        else:
            self.thresholds = thresholds

    def confusion_matrix_at_threshold(self, threshold):
        """Calcula matriz de confusão para um threshold específico"""
        y_pred = (self.y_pred_proba >= threshold).astype(int)

        TP = np.sum((y_pred == 1) & (self.y_true == 1))
        FP = np.sum((y_pred == 1) & (self.y_true == 0))
        TN = np.sum((y_pred == 0) & (self.y_true == 0))
        FN = np.sum((y_pred == 0) & (self.y_true == 1))

        return TP, FP, TN, FN

    def roc_curve_corrected(self):
        """
        Implementação CORRETA da curva ROC seguindo sua descrição:
        - Varia threshold de 1→0
        - Para cada threshold calcula FPR e TPR
        - Conecta os pontos formando degraus
        """
        tpr_list = []  # True Positive Rate (Sensibilidade)
        fpr_list = []  # False Positive Rate (1 - Especificidade)

        print(f"Calculando ROC com {len(self.thresholds)} thresholds...")

        # Ordenar thresholds em ordem DECRESCENTE (de 1.0 a 0.0)
        thresholds_sorted = np.sort(self.thresholds)[::-1]

        for threshold in thresholds_sorted:
            TP, FP, TN, FN = self.confusion_matrix_at_threshold(threshold)

            # Cálculo CORRETO seguindo suas fórmulas:
            # TPR = TP / (TP + FN)
            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0

            # FPR = FP / (FP + TN)
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Garantir que começamos em (0,0) e terminamos em (1,1)
        fpr_list = [0.0] + fpr_list + [1.0]
        tpr_list = [0.0] + tpr_list + [1.0]

        return np.array(fpr_list), np.array(tpr_list), thresholds_sorted

    def auc_roc_corrected(self, fpr, tpr):
        """
        Calcula AUC usando método dos trapézios - CORRIGIDO
        """
        # Ordenar por FPR
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # Calcular área usando regra do trapézio
        area = 0.0
        for i in range(1, len(fpr_sorted)):
            width = fpr_sorted[i] - fpr_sorted[i-1]
            avg_height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2.0
            area += width * avg_height

        return area

    def plot_roc_curve_detailed(self, fpr, tpr, auc_score, save_path=None):
        """Plot detalhado da curva ROC com anotações"""
        plt.figure(figsize=(10, 8))

        # Curva ROC principal
        plt.plot(fpr, tpr, color='darkblue', lw=3, label=f'ROC Curve (AUC = {auc_score:.4f})')

        # Linha de referência (classificador aleatório)
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',
                label='Classificador Aleatório (AUC = 0.5)')

        # Pontos de interesse
        interesting_points = [
            (0.0, 1.0, 'Perfeito', 'green'),
            (0.2, 0.8, 'Bom', 'orange'),
            (0.5, 0.5, 'Aleatório', 'red')
        ]

        for fpr_val, tpr_val, label, color in interesting_points:
            plt.scatter(fpr_val, tpr_val, color=color, s=100, zorder=5, label=label)
            plt.annotate(label, (fpr_val, tpr_val), xytext=(10, 10),
                        textcoords='offset points', fontsize=10)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate (FPR)\n1 - Especificidade', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)\nSensibilidade/Recall', fontsize=12)
        plt.title('CURVA ROC - Análise do Poder Discriminativo do Modelo', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Adicionar explicação
        textstr = '\n'.join([
            'Interpretação:',
            '• (0,1): Classificador Perfeito',
            '• ↗ Curva: Melhor Discriminação',
            '• AUC > 0.9: Excelente',
            '• AUC > 0.8: Bom',
            '• AUC = 0.5: Aleatório'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.6, 0.2, textstr, fontsize=10, verticalalignment='bottom', bbox=props)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curva ROC salva em: {save_path}")

        plt.show()
        return plt

    def generate_roc_analysis_report(self, fpr, tpr, thresholds, auc_score):
        """Gera relatório detalhado da análise ROC"""

        # Encontrar melhor threshold (Youden's J statistic)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_fpr = fpr[best_idx]
        best_tpr = tpr[best_idx]

        report = {
            'auc_roc': auc_score,
            'best_threshold': best_threshold,
            'best_fpr': best_fpr,
            'best_tpr': best_tpr,
            'youden_index': youden_j[best_idx],
            'roc_points': list(zip(fpr, tpr, thresholds))
        }

        return report

    def comprehensive_roc_analysis(self, save_dir=None):
        """Análise completa ROC com todos os outputs"""

        # Calcular curva ROC
        fpr, tpr, thresholds = self.roc_curve_corrected()
        auc_score = self.auc_roc_corrected(fpr, tpr)

        # Gerar relatório
        roc_report = self.generate_roc_analysis_report(fpr, tpr, thresholds, auc_score)

        # Plotar
        if save_dir:
            plot_path = f"{save_dir}/roc_curve_detailed.png"
            self.plot_roc_curve_detailed(fpr, tpr, auc_score, plot_path)

        # Salvar dados da curva ROC
        if save_dir:
            roc_data = pd.DataFrame({
                'FPR': fpr,
                'TPR': tpr,
                'Threshold': np.concatenate([[1.0], thresholds, [0.0]])
            })
            roc_data.to_csv(f"{save_dir}/roc_curve_data.csv", index=False)

        return roc_report, fpr, tpr, thresholds

    def feature_importance_permutation(self, model, X, y, n_iterations=5):
        """
        Calcula importância das features por permutação - VERSÃO CORRIGIDA
        """
        # Baseline accuracy
        activations_baseline, _ = model.forward(X)
        y_pred_baseline = (activations_baseline[-1] > 0.5).astype(int)
        baseline_accuracy = np.mean(y_pred_baseline.flatten() == y.flatten())

        feature_importance = np.zeros(X.shape[1])

        print(f"Calculando importância para {X.shape[1]} features...")

        for feature_idx in range(X.shape[1]):
            accuracy_loss = 0

            for iteration in range(n_iterations):
                # Cria cópia dos dados
                X_permuted = X.copy()
                # Permuta apenas a feature atual
                original_values = X_permuted[:, feature_idx].copy()
                np.random.shuffle(X_permuted[:, feature_idx])

                # Faz predição com dados permutados
                activations_perm, _ = model.forward(X_permuted)
                y_pred_perm = (activations_perm[-1] > 0.5).astype(int)

                # Calcula acurácia com feature permutada
                accuracy_perm = np.mean(y_pred_perm.flatten() == y.flatten())
                accuracy_loss += baseline_accuracy - accuracy_perm

                # Restaura valores originais
                X_permuted[:, feature_idx] = original_values

            # Média da perda de acurácia
            feature_importance[feature_idx] = accuracy_loss / n_iterations

            if (feature_idx + 1) % 5 == 0:
                print(f"  Progresso: {feature_idx + 1}/{X.shape[1]} features")

        return feature_importance

    # Alternativa mais simples e robusta
    def feature_importance_simple(self, model, X, y, n_iterations=3):
        """
        Versão simplificada e mais robusta para importância de features
        """
        print("Calculando importância das features...")

        # Predição baseline
        activations, _ = model.forward(X)
        y_pred_baseline = (activations[-1] > 0.5).astype(int)
        baseline_score = np.mean(y_pred_baseline.flatten() == y.flatten())

        importance_scores = []

        for feature_idx in range(X.shape[1]):
            total_score_decrease = 0

            for _ in range(n_iterations):
                X_perturbed = X.copy()

                # Perturba a feature (shuffle)
                np.random.shuffle(X_perturbed[:, feature_idx])

                # Predição com feature perturbada
                activations_pert, _ = model.forward(X_perturbed)
                y_pred_pert = (activations_pert[-1] > 0.5).astype(int)
                perturbed_score = np.mean(y_pred_pert.flatten() == y.flatten())

                total_score_decrease += baseline_score - perturbed_score

            avg_decrease = total_score_decrease / n_iterations
            importance_scores.append(max(avg_decrease, 0))  # Não permitir valores negativos

        return np.array(importance_scores)