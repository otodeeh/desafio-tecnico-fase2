# Relatório técnico - Tech Challenge Fase 2

## 1. Resumo

Este trabalho evolui os modelos de diagnóstico da Fase 1 por meio de otimização
de hiperparâmetros com algoritmo genético e adiciona explicações em linguagem
natural com uma LLM. A solução é educacional e não se destina a uso clínico.

## 2. Dados e baseline

Foi mantido o dataset da Fase 1, com 569 observações, 30 variáveis numéricas e
alvo benigno/maligno. A divisão estratificada usa 12% para teste e
`random_state=6546`, preservando a comparação com o trabalho anterior.

Os baselines são Regressão Logística com padronização e Random Forest. As
métricas são accuracy, precision, recall, F1, ROC AUC e matriz de confusão.

## 3. Algoritmo genético

### 3.1 Codificação

- Regressão Logística: `log10(C)`, `class_weight` e `solver`.
- Random Forest: número de árvores, profundidade, mínimo de amostras para divisão
  e folha, quantidade de features e balanceamento.

### 3.2 Operadores

- seleção por torneio;
- crossover uniforme;
- mutação independente por gene;
- elitismo dos dois melhores indivíduos;
- cache para evitar reavaliar cromossomos repetidos.

### 3.3 Fitness

Cada indivíduo é avaliado por validação cruzada estratificada de cinco folds.
O fitness é `0,65 * recall + 0,35 * F1`, priorizando a redução de falsos
negativos sem ignorar o equilíbrio entre precision e recall.

### 3.4 Experimentos

| Configuração | População | Gerações | Crossover | Mutação | Torneio |
|---|---:|---:|---:|---:|---:|
| Compacta | 10 | 6 | 0,80 | 0,10 | 3 |
| Balanceada | 14 | 8 | 0,85 | 0,20 | 3 |
| Exploratória | 18 | 10 | 0,70 | 0,30 | 4 |

Os resultados efetivos são gerados em `artifacts/results/summary.json` e
`comparison.csv`. As curvas ficam em `artifacts/figures`.

### 3.5 Resultados obtidos

| Modelo | Versão | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| Regressão Logística | Baseline | 98,55% | 100,00% | 96,15% | 98,04% | 99,46% |
| Regressão Logística | GA | 97,10% | 96,15% | 96,15% | 96,15% | 99,82% |
| Random Forest | Baseline | 97,10% | 96,15% | 96,15% | 96,15% | 99,46% |
| Random Forest | GA | 95,65% | 92,59% | 96,15% | 94,34% | 99,55% |

Na Regressão Logística, a configuração compacta obteve o maior fitness de CV
(0,9712), com `C=0,04288`, `class_weight=balanced` e `solver=lbfgs`. No Random
Forest, a configuração exploratória venceu com fitness 0,9593, 182 árvores,
profundidade 5 e balanceamento de classes.

Nos 69 casos do teste reservado, ambos os modelos otimizados aumentaram
ligeiramente o ROC AUC, mas não melhoraram accuracy, recall ou F1. A Regressão
Logística baseline permaneceu como melhor modelo de decisão. Isso indica efeito
de teto e variância do conjunto de teste pequeno; escolher novamente com base
nesse teste causaria vazamento. Portanto, o resultado negativo foi preservado e
o baseline é a recomendação para uma eventual etapa posterior de validação.

## 4. Integração com LLM

A LLM recebe apenas dados estruturados derivados do modelo. A Responses API é
usada com um schema Pydantic, impedindo omissão de campos obrigatórios. O prompt
proíbe invenção de sintomas, tratamento e diagnóstico definitivo. A avaliação
usa uma rubrica de fidelidade, clareza, não alucinação, segurança e ação.

## 5. Escalabilidade e monitoramento

A API FastAPI é stateless, emite logs JSON e publica métricas Prometheus. O
Deployment Kubernetes define probes e recursos; o HPA escala entre 2 e 10
réplicas com alvo de 65% de CPU. A implantação em nuvem permanece opcional.

## 6. Testes

Os testes cobrem dados, baseline, limites dos genes, reprodutibilidade, pequeno
experimento genético, evidências, fallback de explicação e endpoint de predição.

## 7. Limitações

- Dataset pequeno e sem validação externa.
- Resultados em um único teste reservado podem variar em outras populações.
- Importância de variável não implica causalidade.
- Explicações da LLM exigem avaliação humana contínua.
- Uso clínico exigiria validação regulatória, privacidade e governança.

## 8. Conclusão

A solução demonstra o ciclo completo exigido: baseline, otimização genética,
comparação, interpretabilidade, LLM, avaliação, monitoramento e escalabilidade.
