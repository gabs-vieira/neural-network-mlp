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


    # 2 - Criar a  Criar MLP: 1 camada oculta com 10 neurônios, saída 1
    mlp = MLP(layer_sizes=[X_np.shape[1], 10, 1], learning_rate=0.05, momentum=0.9)


    # 3 - Treinar os dados
    X_train, X_test, y_train, y_test = mlp.train_mlp_with_split(X_np, y_np, test_size=0.2, epochs=500)


if __name__ == '__main__':
    main()
