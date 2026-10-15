# Interpretabilidade de Modelos de Aprendizado de Máquina (Trabalho I)

Previsão de **recorrência do câncer diferenciado de tireoide** com **KNN**, **Naive Bayes** e **Árvore de Decisão**, e investigação de **por que** cada modelo chega às suas previsões (estrutura da árvore, probabilidades condicionais do NB, vizinhos do KNN, importância por permutação, SHAP e LIME).

## Dataset

**UCI Differentiated Thyroid Cancer Recurrence**: <https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence>

| Requisito | Atendido |
|---|---|
| ≥ 5 features | 15 features usadas |
| Problema de classificação | alvo binário: recorrência (sim/não) |
| Numéricas e categóricas | 1 numérica (idade) + 14 categóricas |
| ≥ 100 instâncias | 383 pacientes |

O script baixa os dados automaticamente para `data/` se eles não existirem.

## Como rodar

```bash
pip install -r requirements.txt
python trabalho_interpretabilidade.py
```

Leva cerca de 3 minutos (a comparação 80/20 × 70/30 é a parte mais lenta). Gera:

- `figuras/`: todos os gráficos usados no relatório e nos slides
- `resultados/`: métricas, hiperparâmetros, tabelas de probabilidade, rankings de importância, regras da árvore, experimento de vazamento e verificações de robustez (`robustez_*.csv`) (`resumo.json` tem os números principais; `log_execucao.txt` tem a saída completa)

## Principais decisões (detalhadas no relatório e nos comentários do código)

- **`resposta_tratamento` removida**: é medida depois do tratamento ("estrutural incompleta" → 89 de 91 recorreram). Usá-la seria vazamento temporal. O script mostra que, com ela, a acurácia sobe para até 95% e ela vira a variável nº 1 nos três modelos.
- 19 linhas idênticas mantidas (todas são o perfil mais comum da base). Sem elas, a acurácia de validação cruzada muda menos de 1,2 ponto.
- Divisão 80/20 estratificada; padronização e escolha de hiperparâmetros feitas só no treino.
- Naive Bayes misto (Normal na idade + categórico com Laplace nas outras 14), implementado no próprio script.
- Hiperparâmetros por validação cruzada de 5 partições (F1 da classe "recorre"), com regra de parcimônia na árvore; curvas de validação e de aprendizado para checar over/underfitting.

## Resultados no teste (77 pacientes, 22 com recorrência)

| Modelo | Acurácia | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Árvore (prof. 2) | 0,883 | 0,955 | 0,824 | 0,945 |
| Naive Bayes | 0,883 | 0,864 | 0,809 | 0,969 |
| KNN (k = 7) | 0,909 | 0,818 | 0,837 | 0,970 |

Acurácia de validação cruzada: 0,86 a 0,90. Treino, validação e teste ficam a menos de 2 pontos de distância nos três modelos.

## Verificações de robustez (Etapa 5 do script)

| Ponto de cuidado | Verificação | Resultado |
|---|---|---|
| NB confiante demais | curva de calibração, Brier, NB calibrado | Brier 0,106 → 0,096; caso exemplo 99,998% → 78% |
| Fumo × sexo | recorrência estratificada por sexo; modelos sem `fumante` | associação nos dois sexos (só 9 mulheres fumantes); acurácia não muda sem a variável |
| Categorias raras | razões do NB com α de 0,1 a 2; IC bootstrap | categorias "puras" dependem de α (risco alto: ×33 a ×641); as demais são estáveis |
| Generalização | 50 divisões, subgrupos, estabilidade da árvore, 80/20 × 70/30 | acurácia média 86–88% (faixa 79–94%); risco na raiz em 98%; 70/30 ≈ 80/20 |

## Estrutura

```
trabalho_interpretabilidade.py   código completo (comentado)
requirements.txt                 dependências
data/                            dataset (baixado automaticamente da UCI se não existir)
figuras/                         gráficos gerados pelo script
resultados/                      métricas, tabelas e verificações geradas pelo script
```

## Autores

Arthur Blasi, Gabriela Roxo e Luan Pacheco. Disciplina de Interpretabilidade dos Modelos de Aprendizado de Máquina, Prof.ª Dr.ª Daniela Oliveira Ferreira do Amaral.
