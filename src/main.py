import pandas as pd
import numpy as np
import random
import math

from data_load import prepare_data
def main():
    print("="*70)
    print("REDE NEURAL MLP - PREVISÃO DE DESEMPENHO NBA")
    print("="*70)


    # 1 - Preparacao dos Dados
    prepare_data()


if __name__ == '__main__':
    main()
