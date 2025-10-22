# results_exporter.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime

class ResultsExporter:
    def __init__(self, base_path="./results"):
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = f"{base_path}/run_{self.timestamp}"

        # Criar diretórios
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(f"{self.results_path}/tables", exist_ok=True)
        os.makedirs(f"{self.results_path}/plots", exist_ok=True)
        os.makedirs(f"{self.results_path}/models", exist_ok=True)

    def save_metrics_table(self, metrics_dict, filename="metrics.csv"):
        """Salva métricas em tabela CSV"""
        df_metrics = pd.DataFrame([metrics_dict])
        df_metrics.to_csv(f"{self.results_path}/tables/{filename}", index=False)
        return df_metrics

    def save_confusion_matrix_plot(self, cm, filename="confusion_matrix.png"):
        """Salva plot da matriz de confusão"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()

        classes = ['Bad (0)', 'Good (1)']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Adicionar valores nas células
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.savefig(f"{self.results_path}/plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

    def save_roc_curve(self, fpr, tpr, auc_score, filename="roc_curve.png"):
        """Salva curva ROC"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"{self.results_path}/plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

    def save_feature_importance(self, importance, feature_names, filename="feature_importance.csv"):
        """Salva importância das features"""
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        df_importance.to_csv(f"{self.results_path}/tables/{filename}", index=False)

        # Plot de importância
        plt.figure(figsize=(10, 8))
        top_features = df_importance.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importância')
        plt.title('Top 15 Variáveis Mais Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{self.results_path}/plots/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        return df_importance

    def save_training_history(self, loss_history, filename="training_history.png"):
        """Salva histórico de treinamento"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Loss')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Histórico de Treinamento - Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.results_path}/plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

    def save_hyperparameter_results(self, results, filename="hyperparameter_results.csv"):
        """Salva resultados do grid search"""
        df_results = pd.DataFrame(results)

        # Expandir dicionário de parâmetros
        params_df = pd.DataFrame([r['params'] for r in results])
        metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'params'} for r in results])

        df_final = pd.concat([params_df, metrics_df], axis=1)
        df_final.to_csv(f"{self.results_path}/tables/{filename}", index=False)

        return df_final

    def generate_summary_report(self, metrics, feature_importance, best_hyperparams=None):
        """Gera relatório sumário em texto"""
        report = f"""
            RELATÓRIO DE ANÁLISE - MODELO NBA PERFORMANCE
            ============================================
            Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            RESUMO DO MODELO:
            ----------------
            Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
            Precisão: {metrics['precision']:.4f}
            Recall: {metrics['recall']:.4f}
            F1-Score: {metrics['f1_score']:.4f}
            AUC-ROC: {metrics['auc_roc']:.4f}

            MATRIZ DE CONFUSÃO:
            ------------------
            TN: {metrics['confusion_matrix'][0,0]} | FP: {metrics['confusion_matrix'][0,1]}
            FN: {metrics['confusion_matrix'][1,0]} | TP: {metrics['confusion_matrix'][1,1]}

            VARIÁVEIS MAIS IMPORTANTES:
            --------------------------
            {feature_importance.head(10).to_string()}

            {'MELHORES HIPERPARÂMETROS:' if best_hyperparams else ''}
            {best_hyperparams if best_hyperparams else ''}

            INTERPRETAÇÃO:
            -------------
            - O modelo apresenta boa acurácia geral ({metrics['accuracy']*100:.1f}%)
            - Poder discriminativo: {'EXCELENTE' if metrics['auc_roc'] > 0.9 else 'BOM' if metrics['auc_roc'] > 0.8 else 'MODERADO'}
            - As variáveis listadas acima têm maior impacto nas previsões

            SUGESTÕES:
            ---------
            1. Coletar mais dados para melhorar generalização
            2. Focar nas variáveis mais importantes para análise
            3. Considerar balanceamento de classes se necessário
            """

        with open(f"{self.results_path}/summary_report.txt", "w", encoding='utf-8') as f:
            f.write(report)

        return report