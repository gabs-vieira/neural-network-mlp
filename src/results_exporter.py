# results_exporter_silent.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime

# Configurar matplotlib para não mostrar plots
plt.switch_backend('Agg')  # Importantíssimo: gera plots sem mostrar

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

        print(f"📁 Pasta de resultados criada: {self.results_path}")

    def save_metrics_table(self, metrics_dict, filename="metrics.csv"):
        """Salva métricas em tabela CSV"""
        df_metrics = pd.DataFrame([metrics_dict])
        filepath = f"{self.results_path}/tables/{filename}"
        df_metrics.to_csv(filepath, index=False)
        print(f"📊 Métricas salvas: {filepath}")
        return df_metrics

    def save_confusion_matrix_plot(self, cm, filename="confusion_matrix.png"):
        """Salva plot da matriz de confusão SEM MOSTRAR"""
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

        filepath = f"{self.results_path}/plots/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Fechar figura para liberar memória
        print(f"📈 Matriz de confusão salva: {filepath}")

    def save_roc_curve(self, fpr, tpr, auc_score, filename="roc_curve.png"):
        """Salva curva ROC SEM MOSTRAR"""
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
            plt.scatter(fpr_val, tpr_val, color=color, s=100, zorder=5)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate (FPR)\n1 - Especificidade', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)\nSensibilidade/Recall', fontsize=12)
        plt.title('CURVA ROC - Análise do Poder Discriminativo', fontsize=14, fontweight='bold')
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

        filepath = f"{self.results_path}/plots/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Curva ROC salva: {filepath}")

    def save_feature_importance(self, importance, feature_names, filename="feature_importance"):
        """Salva importância das features com plot e tabela"""
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        # Salvar tabela
        table_path = f"{self.results_path}/tables/{filename}.csv"
        df_importance.to_csv(table_path, index=False)
        print(f"📊 Importância das features salva: {table_path}")

        # Plot de importância
        plt.figure(figsize=(12, 8))
        top_features = df_importance.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importância')
        plt.title('Top 15 Variáveis Mais Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_path = f"{self.results_path}/plots/{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Plot de importância salvo: {plot_path}")

        return df_importance

    def save_training_history(self, loss_history, filename="training_history.png"):
        """Salva histórico de treinamento SEM MOSTRAR"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Loss', color='blue')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Histórico de Treinamento - Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = f"{self.results_path}/plots/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Histórico de treinamento salvo: {filepath}")

    def save_threshold_analysis(self, metrics_df, best_threshold, best_f1, filename="threshold_analysis.png"):
        """Salva análise de thresholds SEM MOSTRAR"""
        plt.figure(figsize=(12, 8))

        plt.plot(metrics_df['threshold'], metrics_df['accuracy'], 'b-', label='Acurácia', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['precision'], 'g-', label='Precisão', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['recall'], 'r-', label='Recall', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['f1_score'], 'purple', label='F1-Score', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['specificity'], 'orange', label='Especificidade', linewidth=2)

        # Melhor threshold
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                   label=f'Melhor F1 (Threshold={best_threshold:.2f})')

        plt.xlabel('Threshold')
        plt.ylabel('Métrica')
        plt.title('Métricas vs Threshold de Classificação')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        filepath = f"{self.results_path}/plots/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Análise de thresholds salva: {filepath}")

    def save_hyperparameter_results(self, results, filename="hyperparameter_results.csv"):
        """Salva resultados do grid search"""
        df_results = pd.DataFrame(results)

        # Expandir dicionário de parâmetros
        params_df = pd.DataFrame([r['params'] for r in results])
        metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'params'} for r in results])

        df_final = pd.concat([params_df, metrics_df], axis=1)

        filepath = f"{self.results_path}/tables/{filename}"
        df_final.to_csv(filepath, index=False)
        print(f"📊 Resultados de hiperparâmetros salvos: {filepath}")

        return df_final

    def save_comparison_plot(self, metrics_before, metrics_after, filename="model_comparison.png"):
        """Salva comparação entre modelos SEM MOSTRAR"""
        labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
        before = [metrics_before['accuracy'], metrics_before['precision'],
                 metrics_before['recall'], metrics_before['f1_score'], metrics_before['auc_roc']]
        after = [metrics_after['accuracy'], metrics_after['precision'],
                metrics_after['recall'], metrics_after['f1_score'], metrics_after['auc_roc']]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(12, 8))
        plt.bar(x - width/2, before, width, label='Antes do Ajuste')
        plt.bar(x + width/2, after, width, label='Depois do Ajuste')

        plt.xlabel('Métricas')
        plt.ylabel('Score')
        plt.title('Comparação do Modelo Antes e Depois do Ajuste')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Adicionar valores nas barras
        for i, v in enumerate(before):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
        for i, v in enumerate(after):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')

        filepath = f"{self.results_path}/plots/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Comparação de modelos salva: {filepath}")

    def generate_summary_report(self, metrics, feature_importance, best_hyperparams=None):
        """Gera relatório sumário em texto"""
        report = f"""
                RELATÓRIO DE ANÁLISE - MODELO NBA PERFORMANCE
                ============================================
                Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                Pasta: {self.results_path}

                RESUMO DO MODELO:
                ----------------
                Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
                Precisão: {metrics['precision']:.4f}
                Recall: {metrics['recall']:.4f}
                F1-Score: {metrics['f1_score']:.4f}
                AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}

                MATRIZ DE CONFUSÃO:
                ------------------
                TN: {metrics['confusion_matrix'][0,0]} | FP: {metrics['confusion_matrix'][0,1]}
                FN: {metrics['confusion_matrix'][1,0]} | TP: {metrics['confusion_matrix'][1,1]}

                TOP 5 VARIÁVEIS MAIS IMPORTANTES:
                -------------------------------
                """
        for i, row in feature_importance.head().iterrows():
            report += f"{row['Feature']}: {row['Importance']:.4f}\n"

        if best_hyperparams:
            report += f"""
            MELHORES HIPERPARÂMETROS:
            -----------------------
            """
            for k, v in best_hyperparams.items():
                report += f"{k}: {v}\n"

        report += f"""
            INTERPRETAÇÃO:
            -------------
            - Acurácia geral: {metrics['accuracy']*100:.1f}%
            - Poder discriminativo: {'EXCELENTE' if metrics.get('auc_roc', 0) > 0.9 else 'BOM' if metrics.get('auc_roc', 0) > 0.8 else 'MODERADO'}
            - Balanceamento: {'BOM' if abs(metrics['precision'] - metrics['recall']) < 0.2 else 'PODE SER MELHORADO'}

            ARQUIVOS GERADOS:
            ---------------
            /tables/ - Métricas, importância, hiperparâmetros
            /plots/  - Gráficos e visualizações
            /models/ - Modelos treinados (se salvos)

            SUGESTÕES:
            ---------
            1. Focar nas variáveis mais importantes para análise
            2. Considerar balanceamento de classes se necessário
            3. Validar com dados de temporadas diferentes
            """

        report_path = f"{self.results_path}/summary_report.txt"
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(report)

        print(f"📋 Relatório sumário salvo: {report_path}")
        return report

    def save_all_data(self, X_train, X_test, y_train, y_test, feature_names):
        """Salva todos os dados brutos para referência"""
        # Dados de treino/teste
        pd.DataFrame(X_train).to_csv(f"{self.results_path}/tables/X_train.csv", index=False)
        pd.DataFrame(X_test).to_csv(f"{self.results_path}/tables/X_test.csv", index=False)
        pd.DataFrame(y_train).to_csv(f"{self.results_path}/tables/y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv(f"{self.results_path}/tables/y_test.csv", index=False)

        # Nomes das features
        pd.DataFrame({'feature_names': feature_names}).to_csv(
            f"{self.results_path}/tables/feature_names.csv", index=False
        )

        print("📁 Dados brutos salvos na pasta /tables/")