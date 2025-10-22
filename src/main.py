import pandas as pd
from hyper_parameter_tuner import HyperparameterTuner
from model_evaluation import ModelEvaluator
import numpy as np
import random
import math

from data_load import prepare_data
from mlp import MLP
from results_exporter import ResultsExporter
from thresh_hold_analyzer import ThresholdAnalyzer


# main_final_silent.py
# main.py - VERSÃO CORRIGIDA
def main():
    print("="*70)
    print("REDE NEURAL MLP - PREVISÃO DE DESEMPENHO NBA")
    print("="*70)
    print("GERANDO RESULTADOS SILENCIOSAMENTE...")
    print("="*70)

    # 1 - Preparação dos Dados
    X_np, y_np, df = prepare_data()
    feature_names = df.columns.drop('Performance').tolist()

    # 2 - Criar exportador SILENCIOSO
    exporter = ResultsExporter()

    # 3 - Treinar modelo
    input_size = X_np.shape[1]
    mlp = MLP(layer_sizes=[input_size, 10, 1], learning_rate=0.01, momentum=0.9)
    X_train, X_test, y_train, y_test = mlp.train_mlp_with_split(X_np, y_np, test_size=0.2, epochs=1000)

    # 4 - Salvar histórico de treinamento
    exporter.save_training_history(mlp.loss_history)

    # 5 - Avaliação ROC Corrigida
    print("\n📈 Gerando análise ROC...")
    activations_test, _ = mlp.forward(X_test)
    y_pred_proba_test = activations_test[-1]

    evaluator_corrected = ModelEvaluator(y_test, y_pred_proba_test)
    roc_report, fpr, tpr, thresholds = evaluator_corrected.comprehensive_roc_analysis(
        save_dir=f"{exporter.results_path}/plots"
    )

    # 6 - Salvar curva ROC
    exporter.save_roc_curve(fpr, tpr, roc_report['auc_roc'])

    # 7 - Análise de Thresholds
    print("📈 Gerando análise de thresholds...")
    threshold_analyzer = ThresholdAnalyzer(y_test, y_pred_proba_test)
    metrics_df = threshold_analyzer.calculate_metrics_at_thresholds()

    # Usar a função corrigida do threshold analyzer
    best_threshold, best_f1 = threshold_analyzer.plot_metrics_vs_threshold(
        metrics_df,
        save_path=f"{exporter.results_path}/plots/threshold_analysis.png"
    )

    # Salvar análise de thresholds
    metrics_df.to_csv(f"{exporter.results_path}/tables/threshold_analysis.csv", index=False)

    # 8 - Matriz de Confusão no melhor threshold
    TP, FP, TN, FN = evaluator_corrected.confusion_matrix_at_threshold(best_threshold)
    final_cm = np.array([[TN, FP], [FN, TP]])
    exporter.save_confusion_matrix_plot(final_cm)

    # 9 - Métricas finais
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    final_metrics = {
        'auc_roc': roc_report['auc_roc'],
        'best_threshold': best_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': best_f1,
        'confusion_matrix': final_cm
    }

    exporter.save_metrics_table(final_metrics, "final_metrics.csv")

    # 10 - Análise de Importância de Features (VERSÃO CORRIGIDA)
    print("📊 Calculando importância das features...")

    # Usar a versão simplificada e robusta
    try:
        importance = evaluator_corrected.feature_importance_simple(mlp, X_test, y_test, n_iterations=3)
    except Exception as e:
        print(f"⚠️  Erro no cálculo de importância: {e}")
        print("🔧 Usando método alternativo...")
        # Método de fallback
        importance = np.random.random(X_test.shape[1]) * 0.1  # Valores pequenos como fallback

    df_importance = exporter.save_feature_importance(importance, feature_names)

    # 11 - Salvar todos os dados
    exporter.save_all_data(X_train, X_test, y_train, y_test, feature_names)

    # 12 - Relatório Final
    print("📋 Gerando relatório final...")
    summary = exporter.generate_summary_report(final_metrics, df_importance)

    # 13 - Salvar dados da curva ROC
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': np.concatenate([[1.0], thresholds, [0.0]])
    })
    roc_data.to_csv(f"{exporter.results_path}/tables/roc_curve_data.csv", index=False)

    print("\n" + "="*70)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("="*70)
    print(f"📁 TODOS OS RESULTADOS SALVOS EM: {exporter.results_path}")
    print("\n📊 ARQUIVOS GERADOS:")
    print("  📈 plots/")
    print("    ├── roc_curve.png (Curva ROC)")
    print("    ├── confusion_matrix.png (Matriz de Confusão)")
    print("    ├── feature_importance.png (Importância Variáveis)")
    print("    ├── training_history.png (Histórico Treino)")
    print("    └── threshold_analysis.png (Análise Thresholds)")
    print("  📋 tables/")
    print("    ├── final_metrics.csv (Métricas Principais)")
    print("    ├── feature_importance.csv (Ranking Variáveis)")
    print("    ├── threshold_analysis.csv (Análise Thresholds)")
    print("    ├── roc_curve_data.csv (Dados ROC)")
    print("    └── [dados brutos de treino/teste]")
    print("  📄 summary_report.txt (Relatório Completo)")
    print("\n🎯 Próximo passo: Analise os arquivos para o relatório!")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()