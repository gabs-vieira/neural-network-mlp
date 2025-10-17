# -*- coding: utf-8 -*-
"""
Script principal para treinar a rede neural com dados da NBA
Executa todo o processo: limpeza → treinamento → avaliação
"""

import pandas as pd
import numpy as np
import random
import math
from rede_neural import RedeNeural


def carregar_dados_limpos(caminho):
    """Carrega os dados já limpos e normalizados usando Pandas"""
    print(f"Carregando dados limpos: {caminho}")
    
    # Carregar com Pandas
    df = pd.read_csv(caminho)
    
    print(f"  - {len(df)} exemplos carregados")
    print(f"  - {len(df.columns)} features por exemplo")
    
    # Converter para listas (compatível com a rede neural)
    cabecalho = list(df.columns)
    dados = df.values.tolist()
    
    return cabecalho, dados


def preparar_dados_para_rede(cabecalho, dados, nome_target='PTS'):
    """
    Separa os dados em X (features) e y (target)
    
    nome_target: nome da coluna que queremos prever
    """
    print(f"\nPreparando dados para a rede neural...")
    print(f"Target escolhido: {nome_target}")
    
    # Encontrar índice da coluna target
    try:
        indice_target = cabecalho.index(nome_target)
    except ValueError:
        print(f"ERRO: Coluna '{nome_target}' não encontrada!")
        print(f"Colunas disponíveis: {cabecalho}")
        return None, None
    
    # Separar features (X) e target (y)
    X = []
    y = []
    
    for linha in dados:
        # Features: todas as colunas exceto o target
        features = []
        for i, valor in enumerate(linha):
            if i != indice_target:
                features.append(valor)
        
        # Target: a coluna que queremos prever
        target = linha[indice_target]
        
        X.append(features)
        y.append([target])  # Lista porque a rede espera lista
    
    print(f"  - Features por exemplo: {len(X[0])}")
    print(f"  - Exemplos totais: {len(X)}")
    
    return X, y


def dividir_treino_teste(X, y, proporcao_teste=0.2):
    """
    Divide os dados em treino e teste
    
    proporcao_teste: % dos dados para teste (ex: 0.2 = 20%)
    """
    print(f"\nDividindo dados: {int((1-proporcao_teste)*100)}% treino, {int(proporcao_teste*100)}% teste")
    
    # Embaralhar dados
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calcular ponto de divisão
    ponto_divisao = int(len(X) * (1 - proporcao_teste))
    
    # Dividir
    X_treino = [X[i] for i in indices[:ponto_divisao]]
    y_treino = [y[i] for i in indices[:ponto_divisao]]
    
    X_teste = [X[i] for i in indices[ponto_divisao:]]
    y_teste = [y[i] for i in indices[ponto_divisao:]]
    
    print(f"  - Treino: {len(X_treino)} exemplos")
    print(f"  - Teste: {len(X_teste)} exemplos")
    
    return X_treino, y_treino, X_teste, y_teste


def calcular_metricas(y_real, y_previsto):
    """
    Calcula métricas de avaliação
    """
    n = len(y_real)
    
    # MSE (Erro Quadrático Médio)
    mse = 0.0
    for i in range(n):
        erro = y_real[i][0] - y_previsto[i][0]
        mse += erro ** 2
    mse = mse / n
    
    # RMSE (Raiz do MSE)
    rmse = math.sqrt(mse)
    
    # MAE (Erro Absoluto Médio)
    mae = 0.0
    for i in range(n):
        erro = abs(y_real[i][0] - y_previsto[i][0])
        mae += erro
    mae = mae / n
    
    # R² (Coeficiente de Determinação)
    # Calcular média dos valores reais
    media_real = sum(y[0] for y in y_real) / n
    
    # Soma dos quadrados totais
    ss_tot = sum((y[0] - media_real) ** 2 for y in y_real)
    
    # Soma dos quadrados dos resíduos
    ss_res = sum((y_real[i][0] - y_previsto[i][0]) ** 2 for i in range(n))
    
    # R²
    if ss_tot != 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def mostrar_exemplos_previsao(y_real, y_previsto, num_exemplos=10):
    """Mostra alguns exemplos de previsões"""
    print(f"\nExemplos de previsões:")
    print("-" * 50)
    print(f"{'Real':>15} | {'Previsto':>15} | {'Erro':>15}")
    print("-" * 50)
    
    for i in range(min(num_exemplos, len(y_real))):
        real = y_real[i][0]
        prev = y_previsto[i][0]
        erro = abs(real - prev)
        
        print(f"{real:15.4f} | {prev:15.4f} | {erro:15.4f}")
    
    print("-" * 50)


def main():
    print("="*70)
    print("REDE NEURAL MLP - PREVISÃO DE DESEMPENHO NBA")
    print("="*70)
    
    # Configurações
    random.seed(42)  # Para resultados reproduzíveis
    
    # 1. Verificar se dados limpos existem
    arquivo_limpo_minmax = '../data/nba_dados_limpos_minmax.csv'
    arquivo_limpo_standard = '../data/nba_dados_limpos_standard.csv'
    
    # Tentar carregar arquivo MinMax primeiro (melhor para redes neurais)
    try:
        print("Tentando carregar dados normalizados com MinMax...")
        cabecalho, dados = carregar_dados_limpos(arquivo_limpo_minmax)
        print("✅ Dados MinMax carregados com sucesso!")
    except FileNotFoundError:
        try:
            print("Arquivo MinMax não encontrado. Tentando Standard...")
            cabecalho, dados = carregar_dados_limpos(arquivo_limpo_standard)
            print("✅ Dados Standard carregados com sucesso!")
        except FileNotFoundError:
            print(f"ERRO: Nenhum arquivo de dados limpos encontrado!")
            print(f"Arquivos procurados:")
            print(f"  - {arquivo_limpo_minmax}")
            print(f"  - {arquivo_limpo_standard}")
            print("\nPrimeiro execute: python limpar_dados.py")
            return
    
    # 2. Preparar dados para a rede
    X, y = preparar_dados_para_rede(cabecalho, dados, nome_target='PTS')
    
    if X is None:
        return
    
    # 3. Dividir em treino e teste
    X_treino, y_treino, X_teste, y_teste = dividir_treino_teste(X, y, proporcao_teste=0.2)
    
    # 4. Criar e treinar a rede neural
    print("\n" + "="*70)
    print("CRIANDO E TREINANDO A REDE NEURAL")
    print("="*70)
    
    # Arquitetura da rede
    num_entradas = len(X_treino[0])
    arquitetura = [num_entradas, 16, 8, 1]  # Entrada → 16 → 8 → 1 saída
    
    print(f"Arquitetura da rede: {arquitetura}")
    
    # Criar rede
    rede = RedeNeural(
        arquitetura=arquitetura,
        taxa_aprendizado=0.1
    )
    
    # Treinar rede
    rede.treinar(
        dados_X=X_treino,
        dados_y=y_treino,
        epocas=800,
        mostrar_progresso=True
    )
    
    # 5. Avaliar no conjunto de teste
    print("\n" + "="*70)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*70)
    
    # Fazer previsões
    y_previsto = []
    for x in X_teste:
        previsao = rede.prever(x)
        y_previsto.append(previsao)
    
    # Calcular métricas
    metricas = calcular_metricas(y_teste, y_previsto)
    
    # Mostrar resultados
    print(f"\nMétricas de Avaliação:")
    print(f"  MSE (Erro Quadrático Médio):     {metricas['MSE']:.6f}")
    print(f"  RMSE (Raiz do MSE):              {metricas['RMSE']:.6f}")
    print(f"  MAE (Erro Absoluto Médio):       {metricas['MAE']:.6f}")
    print(f"  R² (Coeficiente de Determinação): {metricas['R2']:.6f}")
    
    # Interpretar R²
    print(f"\nInterpretação do R²:")
    if metricas['R2'] > 0.8:
        print("  🟢 EXCELENTE! O modelo explica mais de 80% da variação.")
    elif metricas['R2'] > 0.6:
        print("  🟡 BOM! O modelo explica mais de 60% da variação.")
    elif metricas['R2'] > 0.4:
        print("  🟠 MODERADO. O modelo explica mais de 40% da variação.")
    else:
        print("  🔴 BAIXO. O modelo precisa de melhorias.")
    
    print(f"  O modelo explica {metricas['R2']*100:.1f}% da variação nos dados.")
    
    # Mostrar exemplos
    mostrar_exemplos_previsao(y_teste, y_previsto, num_exemplos=15)
    
    # 6. Resumo final
    print("\n" + "="*70)
    print("RESUMO DO TREINAMENTO")
    print("="*70)
    print(f"Dados utilizados:        {len(dados)} jogadores")
    print(f"Features por jogador:    {len(X[0])}")
    print(f"Exemplos de treino:      {len(X_treino)}")
    print(f"Exemplos de teste:       {len(X_teste)}")
    print(f"Arquitetura da rede:     {arquitetura}")
    print(f"Épocas de treinamento:   800")
    print(f"R² final:                {metricas['R2']:.4f}")
    print(f"RMSE final:              {metricas['RMSE']:.4f}")
    
    print("\n✅ Processo concluído com sucesso!")


if __name__ == '__main__':
    main()
