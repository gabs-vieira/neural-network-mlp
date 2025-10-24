import pandas as pd
from model_evaluation import ModelEvaluator
import numpy as np

from data_load import prepare_data
from mlp import MLP
from results_exporter import ResultsExporter
from thresh_hold_analyzer import ThresholdAnalyzer
from hyper_parameter_tuner import HyperparameterTuner


def main():
    print("="*70)
    print("REDE NEURAL MLP - PREVISÃO DE DESEMPENHO NBA")
    print("="*70)

    # 1 - Preparação dos Dados
    X_np, y_np, df = prepare_data()
    feature_names = df.columns.drop('Performance').tolist()

    exporter = ResultsExporter()

    # 2 - HYPERPARAMETER TUNING
    print("\n" + "="*70)
    print("INICIANDO AJUSTE DE HIPERPARÂMETROS")
    print("="*70)
    
    # Definir grid de parâmetros para testar
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_neurons': [5, 10, 15, 20],
        'momentum': [0.8, 0.9, 0.95]
    }
    
    # Executar grid search
    tuner = HyperparameterTuner(X_np, y_np)
    best_result, all_results = tuner.grid_search(
        param_grid=param_grid, 
        test_size=0.2, 
        epochs=500  # Menos épocas para o tuning
    )
    
    # Salvar resultados do hyperparameter tuning
    exporter.save_hyperparameter_results(all_results, "hyperparameter_results.csv")
    
    # Usar os melhores parâmetros encontrados
    best_params = best_result['params']
    print(f"\nUsando melhores parâmetros: {best_params}")

    # 3 - Criar MLP com melhores parâmetros
    input_size = X_np.shape[1]
    mlp = MLP(
        layer_sizes=[input_size, best_params['hidden_neurons'], 1], 
        learning_rate=best_params['learning_rate'], 
        momentum=best_params['momentum']
    )

    # 4 - Treinar modelo final com mais épocas
    print(f"\n🚀 Treinando modelo final com {best_params}...")
    X_train, X_test, y_train, y_test = mlp.train_mlp_with_split(
        X_np, y_np, test_size=0.2, epochs=1000
    )
    exporter.save_training_history(mlp.loss_history)

    # 5 - Avaliação/ geração dos dados
    #========================
    #INICIO AVALIACAO
    #========================
    print("Gerando análise ROC...")
    activations_test, _ = mlp.forward(X_test)
    y_pred_proba_test = activations_test[-1]

    evaluator_corrected = ModelEvaluator(y_test, y_pred_proba_test)
    roc_report, fpr, tpr, thresholds = evaluator_corrected.comprehensive_roc_analysis(
        save_dir=f"{exporter.results_path}/plots"
    )

    exporter.save_roc_curve(fpr, tpr, roc_report['auc_roc'])
    print("Gerando análise de thresholds...")
    threshold_analyzer = ThresholdAnalyzer(y_test, y_pred_proba_test)
    metrics_df = threshold_analyzer.calculate_metrics_at_thresholds()

    best_threshold, best_f1 = threshold_analyzer.plot_metrics_vs_threshold(
        metrics_df,
        save_path=f"{exporter.results_path}/plots/threshold_analysis.png"
    )

    metrics_df.to_csv(f"{exporter.results_path}/tables/threshold_analysis.csv", index=False)

    # Matriz de Confusão no melhor threshold
    TP, FP, TN, FN = evaluator_corrected.confusion_matrix_at_threshold(best_threshold)
    final_cm = np.array([[TN, FP], [FN, TP]])
    exporter.save_confusion_matrix_plot(final_cm)

    # Métricas finais
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

    # Importância das features
    print("Calculando importância das features...")
    importance = evaluator_corrected.feature_importance_simple(mlp, X_test, y_test, n_iterations=3)
    df_importance = exporter.save_feature_importance(importance, feature_names)

    exporter.save_all_data(X_train, X_test, y_train, y_test, feature_names)

    # Gerar relatório final incluindo hiperparâmetros
    print("Gerando relatório final...")
    summary = exporter.generate_summary_report(final_metrics, df_importance, best_params)

    # Salva os dados da curva ROC
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': np.concatenate([[1.0], thresholds, [0.0]])
    })
    roc_data.to_csv(f"{exporter.results_path}/tables/roc_curve_data.csv", index=False)

    # Gerar comparação antes/depois do tuning
    print("Gerando comparação de modelos...")
    
    # Métricas do modelo com parâmetros padrão (para comparação)
    default_mlp = MLP(layer_sizes=[input_size, 10, 1], learning_rate=0.01, momentum=0.9)
    X_train_default, X_test_default, y_train_default, y_test_default = default_mlp.train_mlp_with_split(
        X_np, y_np, test_size=0.2, epochs=1000
    )
    
    activations_default, _ = default_mlp.forward(X_test_default)
    y_pred_proba_default = activations_default[-1]
    evaluator_default = ModelEvaluator(y_test_default, y_pred_proba_default)
    roc_report_default, _, _, _ = evaluator_default.comprehensive_roc_analysis()
    
    TP_default, FP_default, TN_default, FN_default = evaluator_default.confusion_matrix_at_threshold(0.5)
    accuracy_default = (TP_default + TN_default) / (TP_default + FP_default + TN_default + FN_default)
    precision_default = TP_default / (TP_default + FP_default) if (TP_default + FP_default) > 0 else 0
    recall_default = TP_default / (TP_default + FN_default) if (TP_default + FN_default) > 0 else 0
    f1_default = 2 * (precision_default * recall_default) / (precision_default + recall_default) if (precision_default + recall_default) > 0 else 0
    
    metrics_before = {
        'accuracy': accuracy_default,
        'precision': precision_default,
        'recall': recall_default,
        'f1_score': f1_default,
        'auc_roc': roc_report_default['auc_roc']
    }
    
    metrics_after = final_metrics
    exporter.save_comparison_plot(metrics_before, metrics_after, "model_comparison.png")

    #========================
    #FINAL AVALIACAO
    #========================

    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)
    print(f"📊 Melhores hiperparâmetros: {best_params}")
    print(f"📈 Melhoria na acurácia: {final_metrics['accuracy'] - metrics_before['accuracy']:.4f}")
    print(f"📈 Melhoria no F1-Score: {final_metrics['f1_score'] - metrics_before['f1_score']:.4f}")

if __name__ == '__main__':
    main()