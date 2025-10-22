import pandas as pd
from hyper_parameter_tuner import HyperparameterTuner
from model_evaluation import ModelEvaluator, plot_roc_curve
import numpy as np
import random
import math

from data_load import prepare_data
from mlp import MLP
from results_exporter import ResultsExporter

# main.py - VERSÃO COMPLETA
def main():
    print("="*70)
    print("REDE NEURAL MLP - PREVISÃO DE DESEMPENHO NBA")
    print("="*70)

    # 1 - Preparação dos Dados
    X_np, y_np, df = prepare_data()

    # Obter nomes reais das features (excluindo Performance)
    feature_names = df.columns.drop('Performance').tolist()

    # 2 - Criar exportador de resultados
    exporter = ResultsExporter()

    # 3 - Criar e treinar MLP inicial
    input_size = X_np.shape[1]
    mlp = MLP(layer_sizes=[input_size, 10, 1], learning_rate=0.01, momentum=0.9)
    X_train, X_test, y_train, y_test = mlp.train_mlp_with_split(X_np, y_np, test_size=0.2, epochs=1000)

    # 4 - Salvar histórico de treinamento
    exporter.save_training_history(mlp.loss_history)

    # 5 - Avaliação Completa
    print("\n" + "="*70)
    print("AVALIAÇÃO DO MODELO BASE")
    print("="*70)

    # Predições probabilísticas para teste
    activations_test, _ = mlp.forward(X_test)
    y_pred_proba_test = activations_test[-1]

    evaluator = ModelEvaluator(y_test, y_pred_proba_test)
    metrics = evaluator.comprehensive_report(mlp, X_test)

    # 6 - Salvar métricas e plots
    exporter.save_metrics_table(metrics)
    exporter.save_confusion_matrix_plot(metrics['confusion_matrix'])

    # Curva ROC corrigida
    fpr, tpr, thresholds = evaluator.roc_curve()
    auc_score = evaluator.auc_roc(fpr, tpr)
    metrics['auc_roc'] = auc_score  # Atualizar métrica

    exporter.save_roc_curve(fpr, tpr, auc_score)

    # 7 - Análise de Importância de Features
    print("\n" + "="*70)
    print("ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS")
    print("="*70)

    importance = evaluator.feature_importance_permutation(mlp, X_test, y_test, n_iterations=5)
    df_importance = exporter.save_feature_importance(importance, feature_names)

    print("\nTop 10 variáveis mais importantes:")
    for i, row in df_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['Feature']}: {row['Importance']:.4f}")

    # 8 - Ajuste de Hiperparâmetros (OPCIONAL - comentado se estiver com erro)
    try:
        print("\n" + "="*70)
        print("AJUSTE DE HIPERPARÂMETROS")
        print("="*70)

        param_grid = {
            'learning_rate': [0.001, 0.01, 0.05],
            'hidden_neurons': [5, 10, 15],
            'momentum': [0.8, 0.9, 0.95]
        }

        tuner = HyperparameterTuner(X_np, y_np)
        best_result, all_results = tuner.grid_search(param_grid, epochs=500)

        # Salvar resultados do grid search
        exporter.save_hyperparameter_results(all_results)
        best_hyperparams = best_result['params']

    except Exception as e:
        print(f"Grid search não executado: {e}")
        best_hyperparams = None

    # 9 - Gerar relatório final
    summary = exporter.generate_summary_report(metrics, df_importance, best_hyperparams)
    print("\n" + "="*70)
    print("RELATÓRIO FINAL GERADO")
    print("="*70)
    print(f"Todos os resultados salvos em: {exporter.results_path}")
    print(summary)

    # 10 - Salvar dados brutos para referência
    pd.DataFrame(X_train).to_csv(f"{exporter.results_path}/tables/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{exporter.results_path}/tables/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{exporter.results_path}/tables/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{exporter.results_path}/tables/y_test.csv", index=False)

if __name__ == '__main__':
    main()
