 Overview do Dataset NBA Players Performance

O dataset utilizado neste projeto contém informações estatísticas de jogadores da NBA, com o objetivo de prever o nível de performance de cada atleta com base em seus atributos numéricos.

📊 Estrutura do Dataset

Cada linha representa um jogador e seus respectivos dados estatísticos durante uma temporada.
As colunas incluem características gerais, métricas de desempenho e uma classificação final de performance.

| Coluna              | Descrição                                                         |
| :------------------ | :---------------------------------------------------------------- |
| `Player`            | Nome completo do jogador                                          |
| `Pos`               | Posição do jogador em quadra (ex: C, PF, SG, etc.)                |
| `Age`               | Idade do jogador                                                  |
| `Tm`                | Time atual do jogador (abreviação da franquia)                    |
| `G`                 | Número total de jogos disputados                                  |
| `GS`                | Quantidade de jogos como titular                                  |
| `MP`                | Média de minutos jogados por partida                              |
| `FG`, `FGA`, `FG%`  | Cestas de campo feitas, tentadas e percentual de acerto           |
| `3P`, `3PA`, `3P%`  | Arremessos de três pontos feitos, tentados e percentual de acerto |
| `2P`, `2PA`, `2P%`  | Arremessos de dois pontos feitos, tentados e percentual de acerto |
| `FT`, `FTA`, `FT%`  | Lances livres convertidos, tentados e percentual de acerto        |
| `ORB`, `DRB`, `TRB` | Rebotes ofensivos, defensivos e totais                            |
| `AST`               | Assistências médias por jogo                                      |
| `STL`               | Roubos de bola por jogo                                           |
| `BLK`               | Tocos por jogo                                                    |
| `TOV`               | Erros cometidos (turnovers)                                       |
| `PF`                | Faltas pessoais médias                                            |
| `PTS`               | Pontos por jogo                                                   |
| `Performance`       | **Variável alvo**, categórica (ex: “Good”, “Bad”)                 |



🧹 Pré-processamento dos Dados

Durante a etapa de preparação, foram realizadas as seguintes operações:

Leitura e inspeção dos dados com o pandas;

Conversão de separadores decimais (vírgulas → pontos);

Tratamento de valores ausentes e inconsistências;

Normalização dos atributos numéricos para manter escala uniforme entre as variáveis;

Codificação da variável categórica Performance em formato numérico (por exemplo: Good → 1, Bad → 0).

🎯 Objetivo do Dataset

O objetivo é utilizar essas estatísticas como entrada em uma Rede Neural MLP desenvolvida do zero em Python, capaz de classificar a performance dos jogadores (boa ou ruim) com base em seus desempenhos individuais.