# README — Preparação de Dados NBA (Resumo)

## Carregamento e padronização dos nomes das colunas
- Leitura direta com `pandas.read_csv`.
- Nomes normalizados: minúsculas, espaços/símbolos viram `_` e `%` vira `_pct` (ex.: `FG%` → `fg_pct`).
- **Motivo:** evitar bugs ao indexar e facilitar referência no código.

## Conversões para numérico
- Percentuais como `"45%"` → `0.45` (quando a maioria da coluna tem `%`).
- Remoção de ruídos (`,` e espaços) + `to_numeric(errors='coerce')`.
- Tempos `MM:SS` → minutos decimais (ex.: `34:12` → `34.2`).
- **Motivo:** MLP precisa de entradas numéricas; conversões preservam o significado.

## Remoção de duplicatas e colunas não úteis
- Duplicatas removidas (se houver).
- Colunas de identificação/texto livre (ex.: `player`) descartadas.
- **Motivo:** não ajudam a generalizar e podem vazar identidade.

## Tratamento de ausentes (NaN)
- Numéricos: **mediana** por coluna.
- Categóricos: **moda** (ou `"Unknown"` se não houver).
- **Motivo:** estatísticas robustas evitam perder linhas e reduzem viés.

## Codificação de categóricas de baixa cardinalidade
- `get_dummies` em colunas com poucas categorias (ex.: `pos`, `tm`), incluindo coluna para `NaN`.
- **Motivo:** vira entrada numérica sem explosão de dimensionalidade.

## Tratamento de outliers (winsorização suave)
- *Clipping* entre 1º e 99º percentil (apenas em colunas numéricas **não binárias**).
- **Motivo:** reduz a influência de extremos sem remover linhas → treino mais estável.

## Padronização (z-score)
- Para cada coluna numérica (não binária):  
  `x_pad = (x - média) / desvio`
- Parâmetros (média e desvio) salvos em JSON para reuso na inferência.
- **Motivo:** redes treinam melhor com escalas comparáveis e centradas.

## Artefatos gerados
- `*_clean.csv` — dados limpos (numérico + dummies).
- `*_clean_standardized.csv` — versão padronizada (z-score) das colunas numéricas não binárias.
- `*_scaler_params.json` — médias/desvios por coluna.
- `*_prep_report.json` — resumo do processo.
