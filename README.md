# Interpretabilidade de Modelos de Aprendizado de Máquina (Trabalho I)

Previsão de **recorrência do câncer diferenciado de tireoide** com **KNN**, **Naive Bayes** e **Árvore de Decisão**, e investigação de **por que** cada modelo chega às suas previsões: estrutura da árvore, probabilidades condicionais do Naive Bayes, vizinhos do KNN, importância por permutação, SHAP e LIME.

- **Autores:** Arthur Blasi, Gabriela Roxo e Luan Pacheco
- **Disciplina:** Interpretabilidade dos Modelos de Aprendizado de Máquina, Prof.ª Dr.ª Daniela Oliveira Ferreira do Amaral

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

**4. Rodar o projeto**

```bash
python main.py
```

A execução leva cerca de 3 minutos. A parte mais lenta é a comparação entre as divisões 80/20 e 70/30, no fim da execução. Durante a execução, o programa imprime as métricas, as regras da árvore, as probabilidades do Naive Bayes e os resultados de cada verificação.

**5. (Opcional) Rodar os testes**

```bash
python -m unittest discover tests
```

Os testes verificam o Naive Bayes misto implementado no projeto.

**6. Ver os resultados**

Ao terminar, as figuras ficam em `figuras/` e as tabelas em `resultados/`. Se quiser comparar com a execução original, a saída completa do terminal está em `resultados/log_execucao.txt`. A semente aleatória é fixa (42), então os números devem ser os mesmos.

---

## Estrutura do projeto

| Pasta ou arquivo | Função |
|---|---|
| `main.py` | Ponto de entrada. Executa as etapas do trabalho em ordem e salva os resultados. |
| `src/config.py` | Configurações: caminhos, semente, nomes e categorias das colunas, grades de hiperparâmetros e cores. |
| `src/dados.py` | Etapa 1: carrega, traduz e codifica o dataset e faz a divisão treino/teste. |
| `src/modelos.py` | Naive Bayes misto, criação dos três modelos e busca de hiperparâmetros. |
| `src/avaliacao.py` | Métricas, comparação treino × validação × teste e curvas de validação e de aprendizado. |
| `src/interpretabilidade.py` | Etapas 3 e 4: árvore, probabilidades do NB, permutação, SHAP, LIME, caso exemplo e rankings. |
| `src/robustez.py` | Etapa 5: vazamento, linhas idênticas, calibração, fumo × sexo, categorias raras e generalização. |
| `src/graficos.py` | Todas as funções que geram as figuras. |
| `src/saida.py` | Funções para salvar figuras, tabelas e arquivos JSON. |
| `tests/` | Testes unitários do Naive Bayes misto. |
| `requirements.txt` | Bibliotecas necessárias: scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, shap e lime. |
| `data/` | Dataset `Thyroid_Diff.csv` da UCI. Se a pasta não existir, o programa baixa o arquivo automaticamente. |
| `figuras/` | Gráficos gerados (lista abaixo). |
| `resultados/` | Tabelas e números gerados (lista abaixo). |
| `.gitignore` | Arquivos que não vão para o repositório (ambiente virtual e materiais de entrega). |

O `main.py` chama uma função por etapa do enunciado:

| Função em `main.py` | O que faz |
|---|---|
| `dados` | Carrega e traduz o dataset e gera os gráficos exploratórios. |
| `treinamento` | Divide em 80/20, treina os três modelos com validação cruzada, gera as curvas de over/underfitting e avalia no teste. |
| `interpretabilidade` | Árvore (regras e Gini), Naive Bayes (probabilidades condicionais), permutação e SHAP nos três modelos e explicação do caso exemplo (caminho na árvore, contribuições do NB, vizinhos do KNN, SHAP e LIME). |
| `comparacao` | Rankings de importância dos três modelos e correlação entre eles. |
| `robustez` | Experimento de vazamento, linhas idênticas, calibração, fumo × sexo, categorias raras, 50 divisões, subgrupos, estabilidade da árvore e 80/20 × 70/30. |

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

## Decisões e justificativas

| Etapa | Decisão | Justificativa |
|---|---|---|
| Dados | Remover `resposta_tratamento` | É medida depois do tratamento ("estrutural incompleta" → 89 de 91 recorreram). Na hora de prever, ela ainda não existe, então usá-la seria vazamento. Com ela, a acurácia sobe para até 95% e ela vira a variável nº 1 nos três modelos. |
| Dados | Manter as 19 linhas idênticas | Todas são o perfil mais comum da base (mulher jovem, risco baixo, sem recorrência). Com poucos atributos, pacientes diferentes podem ter a mesma ficha. Sem elas, a acurácia de validação muda menos de 1,2 ponto. |
| Dados | Não tratar valores ausentes | O dataset não tem nenhum. |
| Pré-processamento | Codificar as categorias com uma ordem fixa | T, N, estádio e risco seguem a gravidade, o que deixa tabelas e gráficos mais fáceis de ler. |
| Pré-processamento | One-hot nas categóricas (Árvore e KNN) | Evita que o modelo trate categorias como números com distância entre si. |
| Pré-processamento | Padronizar a idade (Árvore e KNN) | O KNN mede distâncias, e sem padronizar a idade (dezenas) pesaria mais que as colunas binárias. A árvore não depende de escala, mas usa o mesmo pré-processamento para facilitar a comparação. |
| Pré-processamento | Naive Bayes misto, implementado no projeto | O `GaussianNB` trataria categorias como números. O NB misto usa Normal na idade e distribuição categórica nas outras 14 variáveis, e suas probabilidades ficam fáceis de ler. |
| Treinamento | Divisão 80/20 estratificada | Mantém 28% de recorrência no treino e no teste. Padronização e hiperparâmetros são ajustados só no treino, para não vazar informação do teste. A divisão 70/30, testada com 30 sementes, deu o mesmo resultado. |
| Treinamento | Hiperparâmetros por validação cruzada de 5 partições, com critério F1 | O F1 da classe "recorre" equilibra encontrar recorrências e evitar alarmes falsos, e não é enganado pelo desbalanceamento. |
| Treinamento | Árvore com profundidade 2 | Entre as configurações a até 1 desvio-padrão da melhor, escolhemos a mais simples. Árvores mais fundas sobem no treino e caem na validação (sem limite: F1 de 1,00 no treino e 0,71 na validação). |
| Treinamento | KNN com k = 7 e pesos uniformes | É o ponto em que as curvas de treino e validação se encontram. Pesos por distância fariam o acerto no treino ser sempre 100%, o que impede de diagnosticar overfitting. |
| Treinamento | Naive Bayes com alpha = 1 | Valores de 0,1 a 2 deram a mesma acurácia. A suavização de Laplace evita probabilidades quase zero nas categorias raras. |
| Avaliação | Acurácia, precisão, recall, especificidade, F1 e ROC-AUC | Como 72% não recorrem, chutar sempre "não" já acerta 71%. O recall é a métrica mais importante, porque deixar passar uma recorrência é o pior erro. |
| Avaliação | Comparar treino, validação e teste e repetir em 50 divisões | Diferenças pequenas indicam que não há overfitting. As 50 divisões mostram quanto o resultado depende da sorte da divisão. |
| Interpretabilidade | Ferramentas próprias de cada modelo | Árvore: desenho, regras e importância de Gini (scikit-learn). Naive Bayes: probabilidades condicionais e decomposição da decisão. KNN: vizinhos mais próximos. |
| Interpretabilidade | Permutação (scikit-learn), SHAP e LIME nos três modelos | Comparar os modelos na mesma régua. A permutação mostra de quais variáveis o desempenho depende, o SHAP divide o crédito entre variáveis correlacionadas e o LIME dá uma explicação local simples. |
| Interpretabilidade | Não usar ELI5 | A permutação do próprio scikit-learn já cumpre o mesmo papel, e o ELI5 não é mantido para as versões atuais do scikit-learn. |

**Bibliotecas usadas:** scikit-learn, pandas, NumPy, Matplotlib e Seaborn, além de SHAP, LIME e SciPy (distribuição Normal do Naive Bayes).

## Resultados no teste (77 pacientes, 22 com recorrência)

| Modelo | Acurácia | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Árvore (profundidade 2) | 0,883 | 0,955 | 0,824 | 0,945 |
| Naive Bayes | 0,883 | 0,864 | 0,809 | 0,969 |
| KNN (k = 7) | 0,909 | 0,818 | 0,837 | 0,970 |

Treino, validação e teste ficam a menos de 2 pontos de distância nos três modelos. Em 50 divisões diferentes, a acurácia média fica entre 86% e 88%.

---

## Análise da interpretabilidade

**Árvore de Decisão.** A árvore escolhida tem profundidade 2 e quatro regras: risco alto → recorre (100% no treino), risco intermediário → recorre (62%), risco baixo e homem → não recorre (80%), risco baixo e mulher → não recorre (96%). O risco responde por 97% da importância de Gini e o sexo pelo restante. Isso faz sentido, porque o risco é uma classificação médica (American Thyroid Association) que já resume tumor, linfonodos e metástase. Em 200 reamostragens, o risco foi a primeira pergunta em 98% das vezes, mas a segunda variável mudou entre amostras.

**Naive Bayes.** As probabilidades condicionais mostram quanto cada característica muda a chance de recorrência: risco alto multiplica por 65, linfonodos laterais (N1b) por 7,5 e sexo masculino por 3,9, enquanto o tumor micropapilífero divide por 17. Como a decisão é uma soma em escala logarítmica, a contribuição de cada variável em cada paciente é exata. O problema é a suposição de independência: risco, T, N e estádio descrevem o mesmo tumor, e o modelo conta essa evidência várias vezes, o que deixa as probabilidades exageradas.

**KNN.** O KNN não tem regras nem pesos para inspecionar. Ele vota com os 7 vizinhos mais parecidos, a distância não diz qual variável aproximou os pacientes e todas as variáveis pesam igual. Com SHAP e LIME, vimos que ele usa o mesmo que os outros modelos: risco, linfonodos (N e adenopatia) e tamanho do tumor (T).

**Caso exemplo.** Um paciente do teste (homem, 80 anos, tumor T4a, sem linfonodos, risco intermediário) teve recorrência, e os três modelos acertaram com confianças diferentes. A árvore deu 62%, porque só olhou o risco. O Naive Bayes deu 99,998%, porque somou T4a, idade e estádio, que já inclui o tumor e a idade. O KNN deu 57%, porque 4 dos 7 vizinhos recorreram: o risco e a idade puxaram para cima e a ausência de linfonodos puxou para baixo. Calibrado, o Naive Bayes cai para 78%.

## Comparação e discussão

**Os resultados fazem sentido em relação ao problema?** Sim. As variáveis mais importantes (risco, linfonodos, tamanho do tumor, metástase e focalidade) são as que a diretriz médica usa para estimar a recorrência, e as direções batem. O único ponto estranho foi o fumo (razão de 4,9 no NB): 82% dos fumantes são homens, e sem essa variável a acurácia não muda, então é uma associação, não uma causa.

**Os três modelos concordam sobre as variáveis mais relevantes?** No topo, sim: o risco é a variável nº 1 nos oito rankings (Gini, permutação, SHAP e razão do NB). Depois disso, a árvore usa só o sexo, enquanto o NB e o KNN distribuem o peso entre linfonodos, adenopatia, T e focalidade. Os três dão a mesma resposta em 87% dos pacientes do teste.

**Quais ferramentas foram usadas em cada modelo?** Árvore: desenho da árvore, regras, caminho do paciente e importância de Gini. Naive Bayes: prioris, probabilidades condicionais, razões de verossimilhança e decomposição da decisão. KNN: lista dos vizinhos mais próximos. Nos três: importância por permutação, SHAP e LIME.

**O que essas ferramentas permitiram compreender?** A estrutura da árvore mostrou que quatro regras baseadas no risco já alcançam 88% de acurácia. As probabilidades do NB mostraram o peso de cada característica e por que ele exagera. Os vizinhos e o SHAP explicaram o KNN. A permutação revelou que N, T e adenopatia são redundantes entre si. E as ferramentas também revelaram o vazamento da resposta ao tratamento, que vira a variável nº 1 nos três modelos quando é usada.

**Quais são as limitações de cada modelo para a interpretabilidade?** A árvore só é legível se for pequena, ignora variáveis informativas e muda com a amostra. O Naive Bayes tem probabilidades exageradas (77% das previsões acima de 99% ou abaixo de 1%) e, nas categorias raras, o valor exato da razão depende da suavização (risco alto vai de ×33 a ×641). O KNN não tem explicação própria, depende de SHAP e LIME, que são aproximações, e seu recall é instável (média de 72% em 50 divisões).

**Desafio final: como a interpretabilidade ajuda a usar as previsões de forma mais consciente?** Ela mostrou que a melhor métrica (95%, com a resposta ao tratamento) vinha de um modelo que olhava o futuro, que modelos com acurácias parecidas usam a informação de formas diferentes e quando desconfiar de uma previsão, como nos 99,998% do Naive Bayes. Uma previsão sozinha ("vai recorrer") não pode ser questionada. Já uma previsão explicada ("vai recorrer porque o tumor é T4a e o paciente tem 80 anos, embora não tenha linfonodos") pode ser verificada, questionada e corrigida por quem decide. A principal limitação do trabalho é que os dados vêm de um único hospital, e o modelo precisaria ser testado em outra população antes de qualquer uso real.
