# Rede Neural MLP para Dados NBA

## 📋 Sobre o Projeto

Este projeto implementa uma **Rede Neural Multicamadas (MLP)** do zero em Python para prever o desempenho de jogadores da NBA. 

**Objetivo:** Prever pontos por jogo (PTS) com base nas outras estatísticas do jogador.

**Tecnologias:** Python puro para a rede neural + Pandas, NumPy e Scikit-learn para processamento de dados.

## 🗂️ Estrutura do Projeto

```
neural-network-mlp/
├── data/
│   ├── nba_dados_2024.csv              # Dados originais (cru)
│   ├── nba_dados_limpos_minmax.csv     # Dados normalizados (0-1)
│   └── nba_dados_limpos_standard.csv   # Dados padronizados (z-score)
│
├── src/
│   ├── limpar_dados.py                 # Script para limpar dados (Pandas + Sklearn)
│   ├── rede_neural.py                  # Implementação da rede neural (Python puro)
│   └── main.py                         # Script principal
│
├── requirements.txt                    # Dependências do projeto
└── README.md                           # Este arquivo
```

## 🚀 Como Executar

### Passo 0: Instalar Dependências
```bash
pip install -r requirements.txt
```

### Passo 1: Limpar os Dados
```bash
cd src
python limpar_dados.py
```

**O que faz:**
- Carrega o arquivo `nba_dados_2024.csv` com **Pandas**
- Remove colunas de texto (Player, Pos, Tm, Performance)
- Converte tudo para números e trata valores faltantes
- Remove jogadores com muitos dados faltando
- Normaliza com **Scikit-learn**:
  - **MinMaxScaler**: valores entre [0, 1] → `nba_dados_limpos_minmax.csv`
  - **StandardScaler**: z-score (média=0, desvio=1) → `nba_dados_limpos_standard.csv`

### Passo 2: Treinar a Rede Neural
```bash
python main.py
```

**O que faz:**
- Carrega os dados limpos
- Divide em 80% treino e 20% teste
- Cria uma rede neural com arquitetura [features, 16, 8, 1]
- Treina por 800 épocas
- Avalia o desempenho no conjunto de teste
- Mostra métricas e exemplos de previsões

## 📊 Exemplo de Resultado

```
MÉTRICAS DE AVALIAÇÃO
=====================
MSE:  0.025000
RMSE: 0.158114
MAE:  0.120000
R²:   0.750000

Interpretação do R²:
🟡 BOM! O modelo explica mais de 60% da variação.
O modelo explica 75.0% da variação nos dados.

Exemplos de previsões:
--------------------------------------------------
           Real |        Previsto |           Erro
--------------------------------------------------
        0.8500 |         0.8234 |         0.0266
        0.3200 |         0.3567 |         0.0367
        0.6100 |         0.5892 |         0.0208
```

## 🧠 Como Funciona a Rede Neural

### 1. Arquitetura
```
Entrada (70 features) → Camada Oculta 1 (16 neurônios) → Camada Oculta 2 (8 neurônios) → Saída (1 neurônio)
```

### 2. Processo de Treinamento

**Forward Propagation:**
1. Dados entram pela camada de entrada
2. Passam pelas camadas ocultas
3. Cada neurônio calcula: `sigmoid(soma_ponderada + bias)`
4. Resultado sai pela camada de saída

**Backpropagation:**
1. Calcula erro entre previsão e valor real
2. Propaga erro de volta pelas camadas
3. Ajusta pesos usando gradiente descendente
4. Repete para todos os exemplos

### 3. Funções Implementadas

**Função de Ativação (Sigmoid):**
```python
f(x) = 1 / (1 + e^(-x))
```
- Transforma qualquer valor em número entre 0 e 1

**Função de Erro (MSE):**
```python
MSE = Σ(valor_real - previsão)² / n
```
- Penaliza erros grandes mais que erros pequenos

## 📈 Métricas de Avaliação

### MSE (Mean Squared Error)
- **O que é:** Média dos erros ao quadrado
- **Interpretação:** Quanto menor, melhor
- **Uso:** Penaliza erros grandes

### RMSE (Root Mean Squared Error)
- **O que é:** Raiz quadrada do MSE
- **Interpretação:** Mesma unidade da variável alvo
- **Uso:** Mais fácil de interpretar que MSE

### MAE (Mean Absolute Error)
- **O que é:** Média dos erros absolutos
- **Interpretação:** Erro médio em termos absolutos
- **Uso:** Não penaliza erros grandes tanto quanto MSE

### R² (Coeficiente de Determinação)
- **O que é:** Proporção da variação explicada pelo modelo
- **Interpretação:** 
  - R² = 1.0: Previsões perfeitas
  - R² = 0.8: Explica 80% da variação (excelente)
  - R² = 0.6: Explica 60% da variação (bom)
  - R² = 0.4: Explica 40% da variação (moderado)
  - R² = 0.0: Não melhor que usar a média
  - R² < 0.0: Pior que usar a média

## 🔧 Configurações Ajustáveis

### No arquivo `main.py`:

**Arquitetura da rede:**
```python
arquitetura = [num_entradas, 16, 8, 1]  # Pode mudar os números
```

**Taxa de aprendizado:**
```python
taxa_aprendizado=0.1  # Valores típicos: 0.01 a 0.5
```

**Número de épocas:**
```python
epocas=800  # Mais épocas = mais treinamento
```

**Divisão treino/teste:**
```python
proporcao_teste=0.2  # 20% para teste, 80% para treino
```

## 🎯 Variável Alvo

Por padrão, o modelo prevê **PTS** (pontos por jogo).

Para mudar a variável alvo, edite esta linha em `main.py`:
```python
X, y = preparar_dados_para_rede(cabecalho, dados, nome_target='PTS')
```

Outras opções possíveis: `'AST'`, `'TRB'`, `'FG%'`, etc.

## 🔍 Interpretação dos Resultados

### Se R² > 0.7:
✅ **Excelente!** O modelo está funcionando muito bem.

### Se R² entre 0.5 e 0.7:
🟡 **Bom.** Resultado satisfatório, mas pode melhorar.

**Como melhorar:**
- Aumentar número de épocas
- Ajustar taxa de aprendizado
- Tentar arquitetura diferente

### Se R² < 0.5:
🔴 **Precisa melhorar.**

**Possíveis problemas:**
- Taxa de aprendizado muito alta ou baixa
- Arquitetura inadequada
- Dados com muito ruído
- Poucas épocas de treinamento

## 💡 Dicas para Melhorar o Modelo

1. **Aumentar épocas:** De 800 para 1500-2000
2. **Ajustar taxa:** Tentar 0.05 ou 0.15
3. **Arquitetura maior:** `[features, 32, 16, 8, 1]`
4. **Mais dados:** Usar mais jogadores se disponível

## 🎓 Conceitos Implementados

- ✅ **Forward Propagation**
- ✅ **Backpropagation** 
- ✅ **Gradiente Descendente**
- ✅ **Função Sigmoid**
- ✅ **Normalização Min-Max**
- ✅ **Métricas de Avaliação (MSE, RMSE, MAE, R²)**
- ✅ **Divisão Treino/Teste**

## 📝 Para o Trabalho da Faculdade

### Pontos a destacar:

1. **Implementação do zero:** Sem bibliotecas de ML
2. **Código limpo:** Bem comentado e fácil de entender
3. **Processo completo:** Limpeza → Treinamento → Avaliação
4. **Métricas adequadas:** MSE, RMSE, MAE, R²
5. **Interpretação:** Explicação clara dos resultados

### Possíveis melhorias para discussão:

- Regularização para evitar overfitting
- Validação cruzada
- Diferentes funções de ativação
- Otimizadores mais avançados (momentum, Adam)
- Feature engineering

## ⚡ Quick Start

```bash
# 1. Limpar dados
cd src
python limpar_dados.py

# 2. Treinar modelo
python main.py
```

**Tempo total:** ~2-3 minutos

**Resultado esperado:** R² entre 0.5 e 0.8

---

**Boa sorte no trabalho! 🚀**
