# -*- coding: utf-8 -*-
"""
Script para limpar e preparar os dados da NBA
Transforma o CSV cru em dados prontos para a rede neural
Usa Pandas, NumPy e Scikit-learn para processamento eficiente
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def carregar_e_limpar_dados(caminho):
    """
    Carrega e faz limpeza inicial dos dados usando Pandas
    """
    print(f"Carregando arquivo: {caminho}")
    
    # Carregar CSV com pandas
    df = pd.read_csv(caminho)
    
    print(f"  - {len(df)} jogadores encontrados")
    print(f"  - {len(df.columns)} colunas no total")
    
    # Mostrar primeiras linhas para debug
    print(f"\nPrimeiras colunas: {list(df.columns[:10])}")
    
    return df


def processar_dados_pandas(df):
    """
    Processa os dados usando Pandas:
    1. Remove colunas de texto
    2. Trata valores faltantes
    3. Converte percentuais
    """
    print("\nProcessando dados com Pandas...")
    
    # Fazer cópia para não modificar original
    df_processado = df.copy()
    
    # Colunas de texto para remover
    colunas_texto = ['Player', 'Pos', 'Tm', 'Performance']
    
    # Remover colunas de texto que existem
    colunas_para_remover = [col for col in colunas_texto if col in df_processado.columns]
    df_processado = df_processado.drop(columns=colunas_para_remover)
    
    print(f"  - Removidas {len(colunas_para_remover)} colunas de texto: {colunas_para_remover}")
    
    # Converter colunas para numérico
    for coluna in df_processado.columns:
        # Limpar valores com vírgulas e aspas
        if df_processado[coluna].dtype == 'object':
            df_processado[coluna] = df_processado[coluna].astype(str)
            df_processado[coluna] = df_processado[coluna].str.replace('"', '')
            df_processado[coluna] = df_processado[coluna].str.replace(',', '.')
        
        # Converter para numérico
        df_processado[coluna] = pd.to_numeric(df_processado[coluna], errors='coerce')
    
    # Tratar valores faltantes (NaN)
    print(f"  - Valores faltantes antes: {df_processado.isnull().sum().sum()}")
    
    # Preencher NaN com 0 (ou poderia usar média/mediana)
    df_processado = df_processado.fillna(0)
    
    print(f"  - Valores faltantes após tratamento: {df_processado.isnull().sum().sum()}")
    
    # Remover linhas onde todas as colunas são zero (jogadores sem dados)
    linhas_antes = len(df_processado)
    df_processado = df_processado[(df_processado != 0).any(axis=1)]
    linhas_depois = len(df_processado)
    
    print(f"  - Removidas {linhas_antes - linhas_depois} linhas vazias")
    print(f"  - {linhas_depois} jogadores restantes")
    print(f"  - {len(df_processado.columns)} colunas numéricas")
    
    return df_processado


def normalizar_com_sklearn(df, metodo='minmax'):
    """
    Normaliza os dados usando Scikit-learn
    
    metodo: 'minmax' (0-1) ou 'standard' (z-score)
    """
    print(f"\nNormalizando dados com Scikit-learn (método: {metodo})...")
    
    if metodo == 'minmax':
        scaler = MinMaxScaler()
        print("  - Usando MinMaxScaler (valores entre 0 e 1)")
    else:
        scaler = StandardScaler()
        print("  - Usando StandardScaler (média=0, desvio=1)")
    
    # Aplicar normalização
    dados_normalizados = scaler.fit_transform(df)
    
    # Criar DataFrame com dados normalizados
    df_normalizado = pd.DataFrame(
        dados_normalizados, 
        columns=df.columns,
        index=df.index
    )
    
    print(f"  - Dados normalizados com sucesso")
    print(f"  - Forma dos dados: {df_normalizado.shape}")
    
    # Mostrar estatísticas
    if metodo == 'minmax':
        print(f"  - Valores mínimos: {df_normalizado.min().min():.3f}")
        print(f"  - Valores máximos: {df_normalizado.max().max():.3f}")
    else:
        print(f"  - Média dos dados: {df_normalizado.mean().mean():.3f}")
        print(f"  - Desvio padrão: {df_normalizado.std().mean():.3f}")
    
    return df_normalizado, scaler


def salvar_dados_pandas(df, caminho_saida):
    """Salva os dados usando Pandas"""
    print(f"\nSalvando dados em: {caminho_saida}")
    
    df.to_csv(caminho_saida, index=False)
    
    print(f"  - {len(df)} linhas salvas")
    print(f"  - {len(df.columns)} colunas salvas")


def main():
    print("="*70)
    print("LIMPEZA DOS DADOS DA NBA COM PANDAS E SCIKIT-LEARN")
    print("="*70)
    
    # Caminhos dos arquivos
    arquivo_entrada = '../data/nba_dados_2024.csv'
    arquivo_saida_minmax = '../data/nba_dados_limpos_minmax.csv'
    arquivo_saida_standard = '../data/nba_dados_limpos_standard.csv'
    
    # Verificar se arquivo existe
    if not os.path.exists(arquivo_entrada):
        print(f"ERRO: Arquivo não encontrado: {arquivo_entrada}")
        return
    
    try:
        # 1. Carregar dados com Pandas
        df_original = carregar_e_limpar_dados(arquivo_entrada)
        
        # 2. Processar dados
        df_processado = processar_dados_pandas(df_original)
        
        # 3. Normalizar com MinMaxScaler (0-1)
        df_minmax, scaler_minmax = normalizar_com_sklearn(df_processado, metodo='minmax')
        
        # 4. Normalizar com StandardScaler (z-score)
        df_standard, scaler_standard = normalizar_com_sklearn(df_processado, metodo='standard')
        
        # 5. Salvar ambas as versões
        salvar_dados_pandas(df_minmax, arquivo_saida_minmax)
        salvar_dados_pandas(df_standard, arquivo_saida_standard)
        
        # 6. Mostrar resumo detalhado
        print("\n" + "="*70)
        print("RESUMO DA LIMPEZA")
        print("="*70)
        print(f"Arquivo original:           {len(df_original)} jogadores")
        print(f"Após processamento:         {len(df_processado)} jogadores")
        print(f"Colunas numéricas:          {len(df_processado.columns)}")
        print(f"\nArquivos gerados:")
        print(f"  MinMax (0-1):             {arquivo_saida_minmax}")
        print(f"  Standard (z-score):       {arquivo_saida_standard}")
        
        # Mostrar algumas estatísticas
        print(f"\nEstatísticas dos dados processados:")
        print(f"  Forma dos dados:          {df_processado.shape}")
        print(f"  Valores únicos por coluna: {df_processado.nunique().mean():.1f} (média)")
        print(f"  Valores zero:             {(df_processado == 0).sum().sum()}")
        
        # Mostrar primeiras colunas
        print(f"\nPrimeiras 10 colunas: {list(df_processado.columns[:10])}")
        
        print("\n✅ Limpeza concluída com sucesso!")
        print("Arquivos prontos para treinar a rede neural.")
        print("\nRecomendação: Use o arquivo MinMax para redes neurais.")
        
    except Exception as e:
        print(f"\n❌ ERRO durante o processamento: {e}")
        print("Verifique se o arquivo CSV está no formato correto.")


if __name__ == '__main__':
    main()
