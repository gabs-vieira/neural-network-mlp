import numpy as np
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, y_true, y_pred_proba, thresholds=None):
        """
        y_true: array com valores reais (0 ou 1)
        y_pred_proba: array com probabilidades preditas [0,1]
        """
        self.y_true = y_true.flatten()
        self.y_pred_proba = y_pred_proba.flatten()

        if thresholds is None:
            self.thresholds = np.arange(0, 1.01, 0.01)
        else:
            self.thresholds = thresholds


    # ===============================
    # Métricas Básicas
    # ===============================
    def confusion_matrix(self, threshold=0.5):
        """Calcula matriz de confusão manualmente"""
        y_pred = (self.y_pred_proba >= threshold).astype(int)

        TP = np.sum((y_pred == 1) & (self.y_true == 1))
        FP = np.sum((y_pred == 1) & (self.y_true == 0))
        TN = np.sum((y_pred == 0) & (self.y_true == 0))
        FN = np.sum((y_pred == 0) & (self.y_true == 1))

        return np.array([[TN, FP], [FN, TP]])

    def accuracy(self, threshold=0.5):
        """Acurácia manual"""
        cm = self.confusion_matrix(threshold)
        return (cm[0,0] + cm[1,1]) / np.sum(cm)

    def precision(self, threshold=0.5):
        """Precisão manual"""
        cm = self.confusion_matrix(threshold)
        TP = cm[1,1]
        FP = cm[0,1]
        return TP / (TP + FP) if (TP + FP) > 0 else 0

    def recall(self, threshold=0.5):
        """Recall manual"""
        cm = self.confusion_matrix(threshold)
        TP = cm[1,1]
        FN = cm[1,0]
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def f1_score(self, threshold=0.5):
        """F1-Score manual"""
        prec = self.precision(threshold)
        rec = self.recall(threshold)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # ===============================
    # Curva ROC e AUC
    # ===============================
    def roc_curve(self):
        """Calcula curva ROC manualmente"""
        tpr_list = []
        fpr_list = []

        for threshold in self.thresholds:
            cm = self.confusion_matrix(threshold)

            # True Positive Rate (Sensibilidade)
            TP = cm[1,1]
            FN = cm[1,0]
            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0

            # False Positive Rate (1 - Especificidade)
            FP = cm[0,1]
            TN = cm[0,0]
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return fpr_list, tpr_list, self.thresholds

    def auc_roc(self, fpr, tpr):
        """Calcula AUC manualmente """
        # Ordena por FPR
        sorted_indices = np.argsort(fpr)
        fpr_sorted = np.array(fpr)[sorted_indices]
        tpr_sorted = np.array(tpr)[sorted_indices]

        # Calcula área usando regra do trapézio
        area = 0.0
        for i in range(1, len(fpr_sorted)):
            width = fpr_sorted[i] - fpr_sorted[i-1]
            height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2.0
            area += width * height

        return max(area, 0.0)  # Garante que não seja negativo

    # ===============================
    # Análise de Importância de Variáveis
    # ===============================
    def feature_importance_permutation(self, model, X, y, n_iterations=10):
        """
        Calcula importância das features por permutação
        """
        baseline_accuracy = self.accuracy()
        feature_importance = np.zeros(X.shape[1])

        for feature_idx in range(X.shape[1]):
            accuracy_loss = 0

            for _ in range(n_iterations):
                # Cria cópia dos dados
                X_permuted = X.copy()
                # Permuta a feature
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

                # Faz predição com dados permutados
                activations, _ = model.forward(X_permuted)
                y_pred_perm = (activations[-1] > 0.5).astype(int)

                # Calcula perda de acurácia
                accuracy_perm = np.mean(y_pred_perm.flatten() == y.flatten())
                accuracy_loss += baseline_accuracy - accuracy_perm

            feature_importance[feature_idx] = accuracy_loss / n_iterations

        return feature_importance

    # ===============================
    # Relatório Completo
    # ===============================
    def comprehensive_report(self, model, X, feature_names=None):
        """Gera relatório completo de avaliação"""
        print("="*70)
        print("RELATÓRIO COMPLETO DO MODELO")
        print("="*70)

        # Métricas básicas
        cm = self.confusion_matrix()
        print(f"\n1. MATRIZ DE CONFUSÃO (threshold=0.5):")
        print(f"   TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
        print(f"   FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")

        print(f"\n2. MÉTRICAS DE PERFORMANCE:")
        print(f"   Acurácia:  {self.accuracy():.4f}")
        print(f"   Precisão:  {self.precision():.4f}")
        print(f"   Recall:    {self.recall():.4f}")
        print(f"   F1-Score:  {self.f1_score():.4f}")

        # Curva ROC
        fpr, tpr, thresholds = self.roc_curve()
        auc_score = self.auc_roc(fpr, tpr)

        print(f"\n3. CURVA ROC:")
        print(f"   AUC-ROC:   {auc_score:.4f}")

        # Encontra melhor threshold (maximizando F1-Score)
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.3, 0.8, 0.05):
            f1 = self.f1_score(threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"\n4. OTIMIZAÇÃO DE THRESHOLD:")
        print(f"   Melhor threshold: {best_threshold:.2f}")
        print(f"   F1-Score no melhor threshold: {best_f1:.4f}")

        return {
            'confusion_matrix': cm,
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score(),
            'auc_roc': auc_score,
            'best_threshold': best_threshold
        }

# ===============================
# Função para Plotar Curva ROC
# ===============================
def plot_roc_curve(fpr, tpr, auc_score, title="Curva ROC"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()