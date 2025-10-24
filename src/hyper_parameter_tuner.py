from mlp import MLP
import numpy as np
from model_evaluation import ModelEvaluator

class HyperparameterTuner:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def grid_search(self, param_grid, test_size=0.2, epochs=500):
        """
        param_grid: dicionário com hiperparâmetros para testar
        Ex: {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_neurons': [5, 10, 15],
            'momentum': [0.8, 0.9, 0.95]
        }
        """
        results = []

        # Gera todas as combinações de parâmetros
        from itertools import product
        param_combinations = list(product(*param_grid.values()))

        print("="*70)
        print("GRID SEARCH - AJUSTE DE HIPERPARÂMETROS")
        print("="*70)

        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_grid.keys(), params))

            print(f"\nTestando combinação {i+1}/{len(param_combinations)}:")
            print(f"  Parâmetros: {param_dict}")

            # Cria e treina modelo
            mlp = MLP(
                layer_sizes=[self.X.shape[1], param_dict['hidden_neurons'], 1],
                learning_rate=param_dict['learning_rate'],
                momentum=param_dict['momentum']
            )

            X_train, X_test, y_train, y_test = mlp.train_mlp_with_split(
                self.X, self.y, test_size=test_size, epochs=epochs
            )

            # Avalia modelo - CORREÇÃO AQUI
            activations, _ = mlp.forward(X_test)
            y_pred_proba = activations[-1]
            
            # Calcular métricas básicas
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_test_flat = y_test.flatten()
            
            # Matriz de confusão
            TP = np.sum((y_pred == 1) & (y_test_flat == 1))
            FP = np.sum((y_pred == 1) & (y_test_flat == 0))
            TN = np.sum((y_pred == 0) & (y_test_flat == 0))
            FN = np.sum((y_pred == 0) & (y_test_flat == 1))
            
            # Métricas
            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # AUC-ROC usando ModelEvaluator
            evaluator = ModelEvaluator(y_test, y_pred_proba)
            fpr, tpr, _ = evaluator.roc_curve_corrected()
            auc_roc = evaluator.auc_roc_corrected(fpr, tpr)

            results.append({
                'params': param_dict,
                'test_accuracy': accuracy,
                'test_f1': f1_score,
                'auc_roc': auc_roc,
                'model': mlp
            })

        # Encontra melhor combinação
        best_result = max(results, key=lambda x: x['test_f1'])

        print("\n" + "="*70)
        print("MELHORES PARÂMETROS ENCONTRADOS:")
        print("="*70)
        print(f"Parâmetros: {best_result['params']}")
        print(f"Acurácia Teste: {best_result['test_accuracy']:.4f}")
        print(f"F1-Score Teste: {best_result['test_f1']:.4f}")
        print(f"AUC-ROC: {best_result['auc_roc']:.4f}")

        return best_result, results