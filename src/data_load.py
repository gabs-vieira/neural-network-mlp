import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data():

    df = pd.read_csv("./../data/nba_dados_2024.csv")

    #Remover colunas irrelevantes
    columns_to_remove = ["Player", "Tm", "Pos"]
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

    #remover todas as linhas que possuirem um valor nulo, visto que todos os dados sao importantes para o modelo
    df = df.dropna()

    #Converter coluna Performance em valor binario
    df["Performance"] = df["Performance"].astype(str).str.strip().str.capitalize()
    df["Performance"] = df["Performance"].map({"Good": 1, "Bad": 0})


    #Substituir virula por ponto em colunas numericas que vieram como strings
    for col in df.columns:
        if col != "Performance" and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    #Separar as nossas variaveis independentes e a dependente -> que vai ser a Performance ja que estamos medindo a performance dos jogadores da NBA
    X = df.drop(columns=["Performance"])
    y = df["Performance"]

    #Normalizacao utilizando StandardScaler (Z-score)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X) # <- calcula (x-media) / desvio padrao

    #Transformar os valores de y em numpy array
    y_np = y.to_numpy(dtype=float).reshape(-1,1) # transformar  y em numpy array
    X_np = X_normalized

    return X_np, y_np, df



