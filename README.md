# Interpretabilidade de Modelos de Aprendizado de Máquina (Trabalho I)

Previsão de **recorrência do câncer diferenciado de tireoide** com **KNN**, **Naive Bayes** e **Árvore de Decisão**, e investigação de **por que** cada modelo chega às suas previsões: estrutura da árvore, probabilidades condicionais do Naive Bayes, vizinhos do KNN, importância por permutação, SHAP e LIME.

**Autores:** Arthur Blasi, Gabriela Roxo e Luan Pacheco
**Disciplina:** Interpretabilidade dos Modelos de Aprendizado de Máquina, Prof.ª Dr.ª Daniela Oliveira Ferreira do Amaral

---

## Passo a passo para rodar

**Pré-requisitos:** Python 3.10 ou superior e Git. A internet só é necessária se a pasta `data/` não existir, porque nesse caso o script baixa o dataset da UCI.

**1. Clonar o repositório**

```bash
git clone https://github.com/gabrielahf/trabalho-interpretabilidade.git
cd trabalho-interpretabilidade
```

**2. Criar e ativar um ambiente virtual (recomendado)**

Linux ou macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**3. Instalar as dependências**

```bash
pip install -r requirements.txt
```

**4. Rodar o script**

```bash
python trabalho_interpretabilidade.py
```

A execução leva cerca de 3 minutos. A parte mais lenta é a comparação entre as divisões 80/20 e 70/30, no fim do script. Durante a execução, o script imprime as métricas, as regras da árvore, as probabilidades do Naive Bayes e os resultados de cada verificação.

**5. Ver os resultados**

Ao terminar, as figuras ficam em `figuras/` e as tabelas em `resultados/`. Se quiser comparar com a execução original, a saída completa do terminal está em `resultados/log_execucao.txt`. A semente aleatória é fixa (42), então os números devem ser os mesmos.

---

## Estrutura do projeto

| Pasta ou arquivo | Função |
|---|---|
| `trabalho_interpretabilidade.py` | Código completo e comentado. Carrega os dados, faz o pré-processamento, treina os três modelos, avalia, gera as análises de interpretabilidade e as verificações de robustez. |
| `requirements.txt` | Bibliotecas necessárias: scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, shap e lime. |
| `data/` | Dataset `Thyroid_Diff.csv` da UCI. Se a pasta não existir, o script baixa o arquivo automaticamente. |
| `figuras/` | Todos os gráficos gerados pelo script (lista abaixo). |
| `resultados/` | Todas as tabelas e números gerados pelo script (lista abaixo). |
| `.gitignore` | Arquivos que não vão para o repositório (ambiente virtual e materiais de entrega). |

O script está dividido nas etapas do enunciado, marcadas por comentários:

| Etapa no código | O que faz |
|---|---|
| Etapa 1: Dados | Baixa e carrega o dataset, traduz as colunas e categorias e gera os gráficos exploratórios. |
| Etapa 2: Pré-processamento e treinamento | Divide em 80/20, codifica as categorias, padroniza a idade, define o Naive Bayes misto, escolhe os hiperparâmetros por validação cruzada e avalia os modelos. |
| Etapa 3a: Árvore de Decisão | Regras, desenho da árvore e importância de Gini. |
| Etapa 3b: Naive Bayes | Prioris, probabilidades condicionais e razões de verossimilhança. |
| Etapa 3c: Ferramentas agnósticas | Importância por permutação e SHAP nos três modelos. |
| Etapa 3d: Explicação local | Caso exemplo: caminho na árvore, decomposição do NB, vizinhos do KNN, SHAP e LIME. |
| Etapa 4: Comparação | Rankings de importância dos três modelos e correlação entre eles. |
| Experimento de vazamento | Mostra o que acontece se a variável "resposta ao tratamento" for usada. |
| Etapa 5: Robustez | Calibração, fumo × sexo, categorias raras, 50 divisões, subgrupos, estabilidade da árvore e 80/20 × 70/30. |

### Arquivos em `figuras/`

| Arquivo | Conteúdo |
|---|---|
| `01_classe_e_idade.png` | Distribuição da classe e da idade por classe |
| `02_taxa_recorrencia_categorias.png` | Taxa de recorrência por categoria (risco, N, adenopatia, T, focalidade, sexo) |
| `04_curvas_validacao.png` | Curvas de validação da árvore (profundidade) e do KNN (k): diagnóstico de over e underfitting |
| `05_curvas_aprendizado.png` | Curvas de aprendizado dos três modelos |
| `06_matrizes_confusao_roc.png` | Matrizes de confusão e curvas ROC no teste |
| `07_arvore.png` | Desenho da árvore de decisão treinada |
| `08_arvore_importancia.png` | Importância das variáveis na árvore (Gini) |
| `09_nb_prob_categoricas.png` | Probabilidades condicionais do Naive Bayes nas 6 variáveis mais importantes |
| `10_nb_gaussiana_idade.png` | Distribuição Normal da idade aprendida pelo Naive Bayes |
| `11_importancia_permutacao.png` | Importância por permutação nos três modelos |
| `12_shap_beeswarm_*.png` | SHAP global de cada modelo (`arvore`, `nb`, `knn`) |
| `13_lime_knn_caso.png` | Explicação LIME do KNN para o caso exemplo |
| `14_shap_waterfall_*.png` | Explicação SHAP de cada modelo para o caso exemplo |
| `15_nb_contribuicoes_caso.png` | Contribuição exata de cada variável no Naive Bayes para o caso exemplo |
| `16_ranking_comparado.png` | Posição de cada variável em 8 rankings de importância |
| `17_spearman_rankings.png` | Correlação entre os rankings |
| `18_shap_global_comparado.png` | Importância SHAP média dos três modelos lado a lado |
| `19_calibracao.png` | Curva de calibração (inclui o Naive Bayes calibrado) |
| `20_fumo_por_sexo.png` | Recorrência por sexo e fumo |
| `21_nb_ic_razoes.png` | Razões de verossimilhança do NB com intervalo de confiança (bootstrap) |
| `22_variacao_50_divisoes.png` | Variação das métricas em 50 divisões treino/teste |

### Arquivos em `resultados/`

| Arquivo | Conteúdo |
|---|---|
| `resumo.json` | Principais números do trabalho em um só lugar |
| `log_execucao.txt` | Saída completa do terminal da última execução |
| `resumo_dados.csv` | Tamanho da base, linhas idênticas e proporção de recorrência |
| `hiperparametros.json` | Hiperparâmetros escolhidos para cada modelo |
| `metricas_teste.csv` | Acurácia, precisão, recall, especificidade, F1 e AUC no teste |
| `treino_cv_teste.csv` | Comparação treino × validação cruzada × teste (overfitting) |
| `sensibilidade_linhas_identicas.csv` | Acurácia com e sem as 19 linhas idênticas |
| `arvore_regras.txt` | Regras da árvore em texto |
| `arvore_importancia_gini.csv` | Importância de Gini por variável |
| `arvore_caminho_caso.txt` | Caminho do caso exemplo na árvore |
| `nb_prob_condicionais_categoricas.csv` | P(categoria \| classe) e razão de verossimilhança de todas as categorias |
| `nb_gaussiana_idade.csv` | Média e desvio da idade em cada classe |
| `nb_importancia_llr.csv` | Importância global de cada variável no NB |
| `nb_contribuicoes_caso.csv` | Contribuição de cada variável no NB para o caso exemplo |
| `importancia_permutacao.csv` | Importância por permutação nos três modelos |
| `importancia_shap.csv` | Importância SHAP média nos três modelos |
| `shap_caso.csv` | Valores SHAP do caso exemplo nos três modelos |
| `lime_caso.json` | Explicação LIME do caso exemplo nos três modelos |
| `knn_vizinhos_caso.csv` | Os 7 vizinhos do caso exemplo no KNN |
| `ranking_variaveis.csv` | Posição de cada variável nos 8 rankings |
| `spearman_rankings.csv` | Correlação de Spearman entre os rankings |
| `experimento_vazamento_resposta.csv` | Métricas dos modelos usando a resposta ao tratamento |
| `robustez.json` | Resumo de todas as verificações de robustez |
| `robustez_1_calibracao.csv` | Brier score e probabilidades extremas, com e sem calibração |
| `robustez_2_fumo_por_sexo.csv` | Recorrência por sexo e fumo |
| `robustez_2_modelos_sem_fumo.csv` | Acurácia e AUC dos modelos sem a variável fumante |
| `robustez_3_razoes_vs_alpha.csv` | Razões do NB para diferentes suavizações |
| `robustez_3_ic_bootstrap.csv` | Intervalo de confiança das razões do NB |
| `robustez_4_50_divisoes.csv` | Média e faixa das métricas em 50 divisões |
| `robustez_4_subgrupos.csv` | Desempenho por sexo e faixa de idade |
| `robustez_5_divisao_80_20_vs_70_30.csv` | Comparação entre as divisões 80/20 e 70/30 (30 sementes cada) |

---

## Dataset

**UCI Differentiated Thyroid Cancer Recurrence**: <https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence>

São 383 pacientes, sem valores ausentes. Usamos 15 variáveis (idade, numérica, e 14 categóricas) para prever a recorrência (sim/não), que acontece em 28% dos casos.

## Principais decisões

- **`resposta_tratamento` removida:** é medida depois do tratamento ("estrutural incompleta" → 89 de 91 recorreram). Usá-la seria vazamento temporal. O script mostra que, com ela, a acurácia sobe para até 95% e ela vira a variável nº 1 nos três modelos.
- **19 linhas idênticas mantidas:** todas são o perfil mais comum da base. Sem elas, a acurácia de validação cruzada muda menos de 1,2 ponto.
- **Divisão 80/20 estratificada:** padronização e escolha de hiperparâmetros feitas só no treino. A divisão 70/30 dá o mesmo resultado.
- **Naive Bayes misto:** Normal na idade e distribuição categórica (com suavização de Laplace) nas outras 14 variáveis, implementado no próprio script.
- **Hiperparâmetros por validação cruzada de 5 partições** (F1 da classe "recorre"), com curvas de validação e de aprendizado para checar over e underfitting.

## Resultados no teste (77 pacientes, 22 com recorrência)

| Modelo | Acurácia | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Árvore (profundidade 2) | 0,883 | 0,955 | 0,824 | 0,945 |
| Naive Bayes | 0,883 | 0,864 | 0,809 | 0,969 |
| KNN (k = 7) | 0,909 | 0,818 | 0,837 | 0,970 |

Treino, validação e teste ficam a menos de 2 pontos de distância nos três modelos. Em 50 divisões diferentes, a acurácia média fica entre 86% e 88%.
