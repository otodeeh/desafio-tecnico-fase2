# Tech Challenge - Fase 2

Otimização dos modelos de diagnóstico desenvolvidos na Fase 1 por meio de um
algoritmo genético, com explicações em linguagem natural produzidas por LLM.

> Projeto educacional. Os resultados não substituem diagnóstico, laudo ou
> avaliação de profissional de saúde.

## Escopo implementado

- Baselines reproduzíveis de Regressão Logística e Random Forest.
- Algoritmo genético próprio com seleção por torneio, crossover uniforme,
  mutação, elitismo e cache de avaliações.
- Três configurações experimentais com diferentes populações e taxas.
- Fitness com validação cruzada: `0,65 * recall + 0,35 * F1`.
- Comparação final no mesmo conjunto de teste reservado.
- Evidências numéricas para cada previsão.
- Integração com OpenAI Responses API e saída estruturada.
- Fallback offline claramente identificado para testes sem chave de API.
- API FastAPI, logs JSON, métricas Prometheus, Docker e Kubernetes HPA.
- Testes automatizados e workflow de integração contínua.

## Relação com a Fase 1

Quando os repositórios estão lado a lado, o carregador encontra automaticamente:

```text
Tech Challenge/
├── desafio-tecnico-fase1/data.csv
└── desafio-tecnico-fase2/
```

Também é possível configurar `PHASE1_DATA_PATH`. Se o CSV não estiver disponível,
o projeto utiliza o dataset equivalente `load_breast_cancer` do scikit-learn.
O conjunto possui 569 observações, 30 variáveis e alvo binário.

## Instalação

```powershell
cd "desafio-tecnico-fase2"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

O arquivo `.env` não deve ser versionado. Para usar a LLM real, preencha
`OPENAI_API_KEY`. O modelo é configurável por `OPENAI_MODEL`.

## Execução

### Notebook

Para a apresentação e a análise completa, abra
`notebooks/tech_challenge_fase2_completo.ipynb`. Ele já está executado e contém:

- análise exploratória do dataset;
- gráficos de distribuição, correlação e PCA;
- comparação dos modelos baseline;
- explicação e resultados dos experimentos com algoritmo genético;
- matrizes de confusão, curvas ROC e importância das variáveis;
- demonstração da explicação em linguagem natural e conclusões do projeto.

O arquivo `notebooks/demo.ipynb` continua disponível como demonstração curta.

Selecione como kernel:

```text
desafio-tecnico-fase2\.venv\Scripts\python.exe
```

No VS Code, use **Select Kernel > Python Environments**. A primeira célula do
notebook também localiza a pasta `src` automaticamente, evitando
`ModuleNotFoundError` quando o pacote ainda não foi instalado em modo editável.
Se estiver disponível na lista, escolha o kernel nomeado
**Python (Tech Challenge Fase 2)**.

Baseline original:

```powershell
python scripts/run_baseline.py
```

Três experimentos de algoritmo genético para os dois modelos:

```powershell
python scripts/run_experiments.py
```

Os resultados são gravados em:

- `artifacts/results/summary.json`
- `artifacts/results/comparison.csv`
- `artifacts/figures/*_convergence.png`
- `artifacts/models/*_best.joblib`

## Resultados verificados

| Modelo | Versão | Accuracy | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|
| Regressão Logística | Baseline | 98,55% | 96,15% | 98,04% | 99,46% |
| Regressão Logística | GA | 97,10% | 96,15% | 96,15% | 99,82% |
| Random Forest | Baseline | 97,10% | 96,15% | 96,15% | 99,46% |
| Random Forest | GA | 95,65% | 96,15% | 94,34% | 99,55% |

O GA melhorou ligeiramente o ROC AUC, mas não superou o baseline nas métricas
de decisão do teste de 69 casos. O resultado foi mantido para evitar seleção
indevida de hiperparâmetros com base no conjunto de teste.

Demonstração da interpretação:

```powershell
python scripts/demo_llm.py
```

Sem `OPENAI_API_KEY`, o comando usa um template offline e informa o provider
`offline-template`. Com a chave configurada, usa a LLM e valida a resposta pelo
schema `ClinicalExplanation`.

API:

```powershell
uvicorn tech_challenge_fase2.api:app --reload
```

- Documentação OpenAPI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Readiness do modelo: `http://localhost:8000/ready`
- Métricas: `http://localhost:8000/metrics`

## Testes

```powershell
pytest --cov=tech_challenge_fase2 --cov-report=term-missing
```

## Metodologia experimental

O conjunto de teste é separado antes da otimização e não participa do fitness.
Cada indivíduo é avaliado por validação cruzada estratificada. Recall recebe
maior peso porque um falso negativo maligno é especialmente indesejado. Após os
três experimentos, apenas o melhor conjunto de hiperparâmetros é treinado em todo
o conjunto de treino e avaliado uma única vez no teste reservado.

## Documentação

- [Apresentação em PDF](output/pdf/Tech_Challenge_Fase2_Apresentacao.pdf)
- [Arquitetura](docs/architecture.md)
- [Relatório técnico](docs/relatorio-tecnico.md)
- [Avaliação da LLM](docs/avaliacao-llm.md)

## Segurança e limitações

- Não envie nome, documento, prontuário ou qualquer dado identificável à LLM.
- A LLM recebe somente a classe, probabilidade e evidências calculadas.
- A LLM não participa da predição e não pode alterar seu resultado.
- Os arquivos Joblib devem ser carregados somente quando produzidos por este
  projeto; artefatos serializados de origem desconhecida não são confiáveis.
- Métricas altas neste dataset pequeno não comprovam generalização clínica.
- Qualquer uso real exigiria validação externa, governança, privacidade e revisão
  por profissionais de saúde.
