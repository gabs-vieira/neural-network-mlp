# -*- coding: utf-8 -*-
"""
Rede Neural MLP implementada do zero
Simples e fácil de entender
"""

import math
import random


def sigmoid(x):
    """Função sigmoid: converte qualquer número em valor entre 0 e 1"""
    if x > 500:  # Evitar overflow
        return 1.0
    elif x < -500:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-x))


def derivada_sigmoid(y):
    """Derivada da sigmoid (para backpropagation)"""
    return y * (1.0 - y)


class RedeNeural:
    """
    Rede Neural Multicamadas (MLP)
    
    Exemplo de uso:
        rede = RedeNeural([10, 5, 1])  # 10 entradas, 5 ocultos, 1 saída
        rede.treinar(dados_X, dados_y, epocas=1000)
        resultado = rede.prever([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    """
    
    def __init__(self, arquitetura, taxa_aprendizado=0.1):
        """
        Cria a rede neural
        
        arquitetura: lista com número de neurônios por camada
                    Ex: [10, 5, 1] = 10 entradas, 5 ocultos, 1 saída
        taxa_aprendizado: velocidade de aprendizado (0.01 a 0.5)
        """
        self.arquitetura = arquitetura
        self.taxa_aprendizado = taxa_aprendizado
        self.num_camadas = len(arquitetura)
        
        # Inicializar pesos aleatoriamente
        self.pesos = []
        self.bias = []
        
        for i in range(self.num_camadas - 1):
            # Pesos entre camada i e i+1
            camada_pesos = []
            for j in range(arquitetura[i]):
                neuronio_pesos = []
                for k in range(arquitetura[i + 1]):
                    # Peso aleatório pequeno
                    peso = random.uniform(-0.5, 0.5)
                    neuronio_pesos.append(peso)
                camada_pesos.append(neuronio_pesos)
            self.pesos.append(camada_pesos)
            
            # Bias para cada neurônio da próxima camada
            camada_bias = []
            for j in range(arquitetura[i + 1]):
                bias = random.uniform(-0.5, 0.5)
                camada_bias.append(bias)
            self.bias.append(camada_bias)
        
        print(f"Rede criada: {arquitetura}")
        print(f"Taxa de aprendizado: {taxa_aprendizado}")
    
    
    def forward(self, entrada):
        """
        Propagação para frente
        Calcula a saída da rede para uma entrada
        """
        ativacoes = [entrada[:]]  # Cópia da entrada
        
        # Para cada camada
        for camada in range(self.num_camadas - 1):
            ativacao_atual = ativacoes[-1]
            proxima_ativacao = []
            
            # Para cada neurônio da próxima camada
            for neuronio in range(self.arquitetura[camada + 1]):
                # Calcular soma ponderada
                soma = self.bias[camada][neuronio]
                
                for entrada_idx in range(len(ativacao_atual)):
                    soma += ativacao_atual[entrada_idx] * self.pesos[camada][entrada_idx][neuronio]
                
                # Aplicar função de ativação
                saida = sigmoid(soma)
                proxima_ativacao.append(saida)
            
            ativacoes.append(proxima_ativacao)
        
        return ativacoes
    
    
    def backward(self, ativacoes, saida_esperada):
        """
        Retropropagação (backpropagation)
        Calcula os erros e gradientes
        """
        # Calcular erro da camada de saída
        saida_rede = ativacoes[-1]
        erros = []
        
        # Erro da última camada
        erro_saida = []
        for i in range(len(saida_rede)):
            erro = saida_esperada[i] - saida_rede[i]
            erro_saida.append(erro)
        erros.append(erro_saida)
        
        # Propagar erro para trás
        for camada in range(self.num_camadas - 2, 0, -1):
            erro_camada = []
            
            for neuronio in range(self.arquitetura[camada]):
                erro_neuronio = 0.0
                
                # Somar contribuições dos neurônios da camada seguinte
                for prox_neuronio in range(len(erros[0])):
                    erro_neuronio += erros[0][prox_neuronio] * self.pesos[camada][neuronio][prox_neuronio]
                
                erro_camada.append(erro_neuronio)
            
            erros.insert(0, erro_camada)
        
        # Calcular deltas (gradientes)
        deltas = []
        for i in range(len(erros)):
            delta_camada = []
            for j in range(len(erros[i])):
                # Delta = erro × derivada da função de ativação
                ativacao = ativacoes[i + 1][j]
                delta = erros[i][j] * derivada_sigmoid(ativacao)
                delta_camada.append(delta)
            deltas.append(delta_camada)
        
        return deltas
    
    
    def atualizar_pesos(self, ativacoes, deltas):
        """
        Atualiza os pesos da rede usando gradiente descendente
        """
        for camada in range(self.num_camadas - 1):
            for neuronio_entrada in range(self.arquitetura[camada]):
                for neuronio_saida in range(self.arquitetura[camada + 1]):
                    # Calcular ajuste do peso
                    ativacao_entrada = ativacoes[camada][neuronio_entrada]
                    delta_saida = deltas[camada][neuronio_saida]
                    ajuste = self.taxa_aprendizado * delta_saida * ativacao_entrada
                    
                    # Atualizar peso
                    self.pesos[camada][neuronio_entrada][neuronio_saida] += ajuste
            
            # Atualizar bias
            for neuronio_saida in range(self.arquitetura[camada + 1]):
                delta_saida = deltas[camada][neuronio_saida]
                ajuste = self.taxa_aprendizado * delta_saida
                self.bias[camada][neuronio_saida] += ajuste
    
    
    def treinar(self, dados_X, dados_y, epocas=1000, mostrar_progresso=True):
        """
        Treina a rede neural
        
        dados_X: lista de listas com as entradas
                Ex: [[1,2,3], [4,5,6], [7,8,9]]
        dados_y: lista de listas com as saídas esperadas
                Ex: [[0.1], [0.5], [0.9]]
        epocas: número de vezes que passa por todos os dados
        """
        print(f"\nIniciando treinamento...")
        print(f"Exemplos: {len(dados_X)}")
        print(f"Épocas: {epocas}")
        
        for epoca in range(epocas):
            erro_total = 0.0
            
            # Para cada exemplo de treinamento
            for i in range(len(dados_X)):
                entrada = dados_X[i]
                saida_esperada = dados_y[i]
                
                # Forward pass
                ativacoes = self.forward(entrada)
                
                # Calcular erro
                saida_rede = ativacoes[-1]
                for j in range(len(saida_esperada)):
                    erro = (saida_esperada[j] - saida_rede[j]) ** 2
                    erro_total += erro
                
                # Backward pass
                deltas = self.backward(ativacoes, saida_esperada)
                
                # Atualizar pesos
                self.atualizar_pesos(ativacoes, deltas)
            
            # Mostrar progresso
            if mostrar_progresso and (epoca % (epocas // 10) == 0 or epoca == epocas - 1):
                erro_medio = erro_total / len(dados_X)
                print(f"Época {epoca + 1:4d}/{epocas} - Erro: {erro_medio:.6f}")
        
        print("✅ Treinamento concluído!")
    
    
    def prever(self, entrada):
        """
        Faz uma previsão para uma entrada
        
        entrada: lista com os valores de entrada
        retorna: lista com as previsões
        """
        ativacoes = self.forward(entrada)
        return ativacoes[-1]
    
    
    def avaliar(self, dados_X, dados_y):
        """
        Avalia o desempenho da rede
        Retorna o erro médio quadrático (MSE)
        """
        erro_total = 0.0
        
        for i in range(len(dados_X)):
            entrada = dados_X[i]
            saida_esperada = dados_y[i]
            previsao = self.prever(entrada)
            
            for j in range(len(saida_esperada)):
                erro = (saida_esperada[j] - previsao[j]) ** 2
                erro_total += erro
        
        mse = erro_total / len(dados_X)
        return mse
