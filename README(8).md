# AI Challenge 2026 — Predicting User Return

> Solução desenvolvida para a fase de qualificação do **International Artificial Intelligence Contest for Children and Youth — AI Challenge 2026**, na categoria **University Students — Individual Participation**.

[English version](#english-version)

---

## 🇧🇷 Português

### Sobre o projeto

Este repositório documenta minha solução para a tarefa **“I’ll Come Back Later — Predicting User Return”**, proposta na fase de qualificação do **AI Challenge 2026**.

O objetivo da tarefa era construir um modelo de Machine Learning capaz de prever se um usuário retornaria a um aplicativo em um futuro próximo, utilizando apenas estatísticas agregadas de comportamento.

A tarefa foi tratada como um problema de **classificação binária supervisionada**, avaliado pela métrica **ROC-AUC**.

## Competição

- **Competição:** International Artificial Intelligence Contest for Children and Youth — AI Challenge 2026
- **Categoria:** University Students — Individual Participation
- **Etapa:** Qualification Stage
- **Tarefa:** I’ll Come Back Later — Predicting User Return
- **Tipo de problema:** Classificação binária
- **Métrica:** ROC-AUC
- **Corte de qualificação:** ROC-AUC ≥ 0.60

A pontuação da tarefa era calculada por:

```text
score = max(0, ROC-AUC - 0.60) / (1 - 0.60)
```

## Objetivo

Prever a variável `retention`, onde:

- `1` — o usuário retornará ao aplicativo;
- `0` — o usuário não retornará.

Para submissão, foram utilizados **scores/probabilidades contínuas** da classe positiva, em vez de simplesmente converter as previsões para `0` e `1`, porque a métrica ROC-AUC avalia principalmente a capacidade do modelo de ordenar corretamente exemplos positivos e negativos.

## Dados

O conjunto de treinamento continha as seguintes variáveis:

| Variável | Descrição |
|---|---|
| `id` | Identificador do usuário |
| `sessions_count` | Número de sessões |
| `avg_session_time` | Duração média das sessões |
| `days_since_last_activity` | Dias desde a última atividade |
| `purchases_count` | Número de compras |
| `avg_purchase_value` | Valor médio das compras |
| `active_days` | Número de dias ativos |
| `session_std` | Variabilidade da duração das sessões |
| `is_weekend_user` | Indica maior atividade nos fins de semana |
| `retention` | Variável alvo |

O conjunto de teste possuía as mesmas variáveis de entrada, sem `retention`.

> Os dados originais da competição não são incluídos neste repositório. Consulte a plataforma oficial da competição para obter os arquivos, respeitando seus termos de uso.

## Estratégia

A solução foi desenvolvida de forma iterativa.

### 1. Baselines

Foram avaliados modelos clássicos, incluindo:

- Logistic Regression;
- Random Forest;
- Extra Trees;
- XGBoost;
- LightGBM;
- CatBoost.

A regressão logística serviu como baseline simples, enquanto os modelos baseados em árvores apresentaram melhor capacidade de capturar relações não lineares entre as variáveis.

### 2. Feature Engineering

Foram testadas variáveis derivadas, como:

```text
sessions_per_active
purchases_per_session
purchases_per_active
recency_per_active
session_cv
total_session_time
purchase_total
activity_balance
value_per_time
```

A ideia foi representar relações comportamentais que não estavam explicitamente disponíveis nas variáveis originais.

### 3. Ensembles

As melhores versões combinaram previsões de diferentes modelos, principalmente:

- CatBoost;
- Extra Trees;
- Random Forest;
- LightGBM;
- XGBoost.

Também foram testadas estratégias de:

- probability blending;
- rank blending;
- bagging com diferentes seeds;
- regularização;
- refinamento dos pesos do ensemble.

Como a métrica era ROC-AUC, trabalhar com rankings das previsões foi especialmente útil em alguns experimentos.

## Validação

Durante o desenvolvimento, foram utilizados experimentos com validação cruzada para comparar modelos e reduzir o risco de escolher uma solução baseada apenas em um único split.

O objetivo era maximizar **ROC-AUC**, e não accuracy.

Isso é importante porque, para ROC-AUC, o ranking das previsões tem mais relevância do que um threshold fixo de classificação.

## Resultado

A melhor submissão alcançou:

```text
ROC-AUC = 0.6753332505974485
```

Pontuação convertida na plataforma:

```text
Score ≈ 0.188
```

Como o corte necessário era:

```text
ROC-AUC ≥ 0.60
```

a solução ultrapassou o limite de qualificação e permitiu o avanço para a fase principal da competição.

## Evolução das submissões

Durante os experimentos, diferentes versões foram submetidas e comparadas.

Algumas das versões intermediárias ficaram aproximadamente na faixa de:

```text
ROC-AUC ≈ 0.670 — 0.675
```

Os experimentos mostraram que pequenas alterações no ensemble produziam diferenças relativamente pequenas no leaderboard, sugerindo que os modelos já estavam explorando sinais bastante semelhantes do conjunto de dados.

## Estrutura recomendada do repositório

```text
ai-challenge-user-return/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── user_return_experiments.ipynb
│
├── src/
│   └── train_and_predict.py
│
└── submissions/
    └── README.md
```

Os arquivos de dados originais e submissões privadas podem ser mantidos fora do GitHub.

## Como reproduzir

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/ai-challenge-user-return.git
cd ai-challenge-user-return
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Adicione os dados

Coloque os arquivos da competição em uma pasta local, por exemplo:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

A pasta `data/` deve permanecer fora do versionamento caso os termos da competição não permitam redistribuição.

### 5. Execute o treinamento

```bash
python src/train_and_predict.py
```

ou abra:

```text
notebooks/user_return_experiments.ipynb
```

## Tecnologias utilizadas

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- LightGBM
- XGBoost
- SciPy
- Jupyter / Google Colab

## Principais aprendizados

Este desafio foi útil para praticar:

- classificação binária;
- ROC-AUC;
- validação cruzada;
- feature engineering;
- modelos baseados em árvores;
- ensemble learning;
- rank blending;
- prevenção de overfitting;
- análise de leaderboard;
- criação de submissões reprodutíveis.

Também mostrou que melhorar um modelo competitivo nem sempre significa aumentar sua complexidade. Em vários experimentos, regularização, diversidade entre modelos e combinação adequada das previsões foram mais importantes do que simplesmente adicionar modelos maiores.

## Observação

Este repositório tem finalidade educacional e de portfólio.

Os dados originais pertencem aos organizadores da competição e não são redistribuídos aqui.

---

# English version

## About the project

This repository documents my solution for **“I’ll Come Back Later — Predicting User Return”**, a qualification task from the **AI Challenge 2026**.

The goal was to build a Machine Learning model capable of predicting whether a user would return to a mobile application in the near future using aggregated behavioral statistics.

The task was formulated as a **supervised binary classification problem** and evaluated using **ROC-AUC**.

## Competition

- **Competition:** International Artificial Intelligence Contest for Children and Youth — AI Challenge 2026
- **Category:** University Students — Individual Participation
- **Stage:** Qualification Stage
- **Task:** I’ll Come Back Later — Predicting User Return
- **Problem type:** Binary classification
- **Metric:** ROC-AUC
- **Qualification threshold:** ROC-AUC ≥ 0.60

The task score was calculated as:

```text
score = max(0, ROC-AUC - 0.60) / (1 - 0.60)
```

## Objective

Predict `retention`, where:

- `1` — the user will return;
- `0` — the user will not return.

For submission, continuous scores/probabilities for the positive class were used instead of hard `0/1` predictions because ROC-AUC mainly measures how well the model ranks positive samples above negative ones.

## Dataset

The training dataset contained:

| Feature | Description |
|---|---|
| `id` | User identifier |
| `sessions_count` | Number of sessions |
| `avg_session_time` | Average session duration |
| `days_since_last_activity` | Days since last activity |
| `purchases_count` | Number of purchases |
| `avg_purchase_value` | Average purchase value |
| `active_days` | Number of active days |
| `session_std` | Session duration variability |
| `is_weekend_user` | Indicates stronger weekend activity |
| `retention` | Target variable |

The test dataset contained the same input variables without `retention`.

> The original competition data is not distributed in this repository. Please refer to the official competition platform and its terms of use.

## Approach

The solution was developed iteratively.

### 1. Baselines

Several classical models were evaluated:

- Logistic Regression;
- Random Forest;
- Extra Trees;
- XGBoost;
- LightGBM;
- CatBoost.

Logistic Regression was used as a simple baseline, while tree-based models were better at capturing nonlinear relationships.

### 2. Feature Engineering

Derived behavioral features were tested, including:

```text
sessions_per_active
purchases_per_session
purchases_per_active
recency_per_active
session_cv
total_session_time
purchase_total
activity_balance
value_per_time
```

These features were designed to expose relationships that were not directly represented in the original dataset.

### 3. Ensembles

The strongest experiments combined predictions from models such as:

- CatBoost;
- Extra Trees;
- Random Forest;
- LightGBM;
- XGBoost.

The experiments also included:

- probability blending;
- rank blending;
- bagging with multiple random seeds;
- regularization;
- ensemble weight refinement.

Since the evaluation metric was ROC-AUC, prediction ranking was especially relevant.

## Validation

Cross-validation experiments were used to compare models and reduce dependency on a single train/validation split.

The optimization target was **ROC-AUC**, rather than accuracy.

For ROC-AUC, ranking quality matters more than choosing a fixed classification threshold.

## Result

The best submission achieved:

```text
ROC-AUC = 0.6753332505974485
```

Platform-converted score:

```text
Score ≈ 0.188
```

The required qualification threshold was:

```text
ROC-AUC ≥ 0.60
```

Therefore, the solution successfully exceeded the qualification requirement and enabled progression to the Main Stage.

## Submission history

Several iterations were evaluated during development.

Intermediate submissions were generally around:

```text
ROC-AUC ≈ 0.670 — 0.675
```

The experiments showed that small changes in ensemble weights resulted in relatively small leaderboard differences, suggesting substantial correlation between the strongest models.

## Suggested repository structure

```text
ai-challenge-user-return/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── user_return_experiments.ipynb
│
├── src/
│   └── train_and_predict.py
│
└── submissions/
    └── README.md
```

Original datasets and private submission files can remain outside GitHub.

## Reproduction

### Clone

```bash
git clone https://github.com/YOUR-USERNAME/ai-challenge-user-return.git
cd ai-challenge-user-return
```

### Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add competition data locally

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Keep the `data/` directory outside version control if redistribution is not permitted by the competition terms.

### Run

```bash
python src/train_and_predict.py
```

or open:

```text
notebooks/user_return_experiments.ipynb
```

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- LightGBM
- XGBoost
- SciPy
- Jupyter / Google Colab

## Key takeaways

This task provided practical experience with:

- binary classification;
- ROC-AUC;
- cross-validation;
- feature engineering;
- tree-based models;
- ensemble learning;
- rank blending;
- overfitting control;
- leaderboard analysis;
- reproducible submission pipelines.

It also reinforced that stronger solutions do not always require more complex models. Regularization, model diversity, validation quality and careful blending can matter more than raw model complexity.

## Disclaimer

This repository is intended for educational and portfolio purposes.

The original competition data belongs to the competition organizers and is not redistributed here.
