import pandas as pd
import numpy as np
import random
import math

from data_load import prepare_data
from mlp import MLP

def main():
    print("="*70)
    print("REDE NEURAL MLP - PREVISÃO DE DESEMPENHO NBA")
    print("="*70)

    # 1 - Preparacao dos Dados
    X_np, y_np, df = prepare_data()

    # VERIFICAÇÕES
    print(f"Shape de X_np: {X_np.shape}")
    print(f"Shape de y_np: {y_np.shape}")
    print(f"Valores únicos em y: {np.unique(y_np)}")
    print(f"NaN em X: {np.isnan(X_np).sum()}, NaN em y: {np.isnan(y_np).sum()}")

    # 2 - Criar MLP
    input_size = X_np.shape[1]
    mlp = MLP(layer_sizes=[input_size, 10, 1], learning_rate=0.01, momentum=0.9)

    # 3 - Treinar os dados
    X_train, X_test, y_train, y_test = mlp.train_mlp_with_split(X_np, y_np, test_size=0.2, epochs=1000)

    # 4 - Avaliar
    train_accuracy = mlp.evaluate(X_train, y_train)
    test_accuracy = mlp.evaluate(X_test, y_test)
    print(f"Acurácia Treino: {train_accuracy:.4f}")
    print(f"Acurácia Teste: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
