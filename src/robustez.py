import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.tree import DecisionTreeClassifier

from src.avaliacao import classificar, configurar_modelo, probabilidade_de_recorrencia
from src.config import (ARVORE, COLUNA_ALVO, COLUNA_VAZAMENTO, NAIVE_BAYES, PROPORCAO_TESTE,
                        SEMENTE, VARIAVEIS_CATEGORICAS, VARIAVEIS_NUMERICAS)
from src.dados import (ESQUEMA_PRINCIPAL, DivisaoTreinoTeste, EsquemaVariaveis,
                       codificar_variaveis, rotulos_da_variavel)
from src.interpretabilidade import ArvoreLegivel, variavel_de_origem
from src.modelos import buscar_hiperparametros, criar_modelos, criar_validacao_cruzada

NOME_NB_CALIBRADO = "Naive Bayes calibrado"


def validacao_repetida(repeticoes: int) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=repeticoes, random_state=SEMENTE)


def experimento_vazamento(dataset: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Treina os modelos com a resposta ao tratamento para mostrar o efeito do vazamento."""
    esquema = EsquemaVariaveis(VARIAVEIS_NUMERICAS, VARIAVEIS_CATEGORICAS + [COLUNA_VAZAMENTO])
    X = codificar_variaveis(dataset, esquema)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=PROPORCAO_TESTE, stratify=y, random_state=SEMENTE)
    linhas = []
    for nome, modelo in criar_modelos(esquema).items():
        busca = buscar_hiperparametros(modelo, nome, X_treino.values, y_treino, criar_validacao_cruzada())
        probabilidades = probabilidade_de_recorrencia(busca.best_estimator_, X_teste)
        permutacao = permutation_importance(busca.best_estimator_, X_teste.values, y_teste, scoring="roc_auc",
                                            n_repeats=20, random_state=SEMENTE)
        importancia = pd.Series(permutacao.importances_mean, index=esquema.variaveis).sort_values(ascending=False)
        linhas.append({
            "modelo": nome,
            "acuracia": accuracy_score(y_teste, classificar(probabilidades)),
            "f1": f1_score(y_teste, classificar(probabilidades)),
            "roc_auc": roc_auc_score(y_teste, probabilidades),
            "variavel_mais_importante": importancia.index[0],
            "queda_auc_top1": importancia.iloc[0],
        })
    return pd.DataFrame(linhas).set_index("modelo").round(3)


def sensibilidade_linhas_identicas(modelos: dict, dataset: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Acurácia de validação cruzada (5x5) com e sem as linhas repetidas."""
    dataset_sem_repetidas = dataset.drop_duplicates()
    X_sem_repetidas = codificar_variaveis(dataset_sem_repetidas).values
    y_sem_repetidas = dataset_sem_repetidas[COLUNA_ALVO].values
    validacao = validacao_repetida(5)
    resultado = {}
    for nome, modelo in modelos.items():
        com = cross_validate(clone(modelo), X.values, y, cv=validacao, scoring="accuracy")["test_score"].mean()
        sem = cross_validate(clone(modelo), X_sem_repetidas, y_sem_repetidas, cv=validacao,
                             scoring="accuracy")["test_score"].mean()
        resultado[nome] = {"acc_cv_com_identicas": float(com), "acc_cv_sem_identicas": float(sem)}
    return pd.DataFrame(resultado).T


def criar_nb_calibrado(modelo_nb) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(clone(modelo_nb), method="sigmoid", cv=5)


def probabilidades_fora_da_amostra(modelos: dict, X: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    """Probabilidade de cada paciente dada por um modelo que não o viu no treino (inclui o NB calibrado)."""
    modelos_avaliados = dict(modelos)
    modelos_avaliados[NOME_NB_CALIBRADO] = criar_nb_calibrado(modelos[NAIVE_BAYES])
    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    return {nome: cross_val_predict(clone(modelo), X.values, y, cv=validacao, method="predict_proba")[:, 1]
            for nome, modelo in modelos_avaliados.items()}


def analisar_calibracao(probabilidades: dict, y: np.ndarray) -> pd.DataFrame:
    """Brier score e quanto o modelo acerta quando dá probabilidades extremas (> 99% ou < 1%)."""
    linhas = []
    for nome, prob in probabilidades.items():
        extremas = (prob > .99) | (prob < .01)
        acerto_nas_extremas = (classificar(prob[extremas]) == y[extremas]).mean() if extremas.any() else np.nan
        linhas.append({
            "modelo": nome,
            "brier": brier_score_loss(y, prob),
            "acuracia_oof": accuracy_score(y, classificar(prob)),
            "auc_oof": roc_auc_score(y, prob),
            "pct_prob_extrema": extremas.mean(),
            "acerto_nas_extremas": acerto_nas_extremas,
        })
    return pd.DataFrame(linhas).set_index("modelo").round(3)


def probabilidade_calibrada_do_paciente(modelo_nb, divisao: DivisaoTreinoTeste, paciente: pd.DataFrame) -> float:
    nb_calibrado = criar_nb_calibrado(modelo_nb).fit(divisao.X_treino.values, divisao.y_treino)
    return float(nb_calibrado.predict_proba(paciente.values)[0, 1])


def recorrencia_por_sexo_e_fumo(dataset: pd.DataFrame) -> pd.DataFrame:
    tabela = dataset.groupby(["sexo", "fumante"])[COLUNA_ALVO].agg(taxa="mean", n="size").reset_index()
    tabela["taxa"] = tabela["taxa"].round(3)
    return tabela


def proporcao_de_homens_entre_fumantes(dataset: pd.DataFrame) -> float:
    return float((dataset.loc[dataset["fumante"] == "sim", "sexo"] == "masculino").mean())


def modelos_sem_fumo(modelos: dict, hiperparametros: dict, dataset: pd.DataFrame, X: pd.DataFrame,
                     y: np.ndarray) -> pd.DataFrame:
    """Compara acurácia e AUC (CV 5x5) com e sem a variável 'fumante'."""
    esquema = EsquemaVariaveis(VARIAVEIS_NUMERICAS, [c for c in VARIAVEIS_CATEGORICAS if c != "fumante"])
    X_sem_fumo = codificar_variaveis(dataset, esquema).values
    modelos_base = criar_modelos(esquema)
    validacao = validacao_repetida(5)
    linhas = []
    for nome, modelo in modelos.items():
        com = cross_validate(clone(modelo), X.values, y, cv=validacao, scoring=["accuracy", "roc_auc"])
        sem = cross_validate(configurar_modelo(modelos_base[nome], hiperparametros[nome]), X_sem_fumo, y,
                             cv=validacao, scoring=["accuracy", "roc_auc"])
        linhas.append({"modelo": nome,
                       "acc_com_fumo": com["test_accuracy"].mean(), "acc_sem_fumo": sem["test_accuracy"].mean(),
                       "auc_com_fumo": com["test_roc_auc"].mean(), "auc_sem_fumo": sem["test_roc_auc"].mean()})
    return pd.DataFrame(linhas).set_index("modelo").round(3)


CATEGORIAS_ANALISADAS = [
    ("risco", "alto"), ("M", "M1"), ("T", "T4a"), ("estadio", "IVB"), ("adenopatia", "extensa"),
    ("adenopatia", "bilateral"), ("N", "N1b"), ("fumante", "sim"), ("sexo", "masculino"),
    ("risco", "baixo"), ("patologia", "micropapilífero"),
]


def razoes_das_categorias(naive_bayes) -> dict[str, float]:
    razoes = {}
    for coluna, rotulo in CATEGORIAS_ANALISADAS:
        posicao = ESQUEMA_PRINCIPAL.categoricas.index(coluna)
        codigo = rotulos_da_variavel(coluna).index(rotulo)
        razoes[f"{coluna}={rotulo}"] = naive_bayes.razao_verossimilhanca(posicao, codigo)
    return razoes


def razoes_por_suavizacao(naive_bayes, divisao: DivisaoTreinoTeste) -> pd.DataFrame:
    """Razão de verossimilhança das categorias analisadas para alpha de 0,1 a 2."""
    tabela = pd.DataFrame({
        f"alpha={alpha}": razoes_das_categorias(
            clone(naive_bayes).set_params(alpha=alpha).fit(divisao.X_treino.values, divisao.y_treino))
        for alpha in [0.1, 0.5, 1.0, 2.0]})
    pacientes = [(divisao.X_treino[c] == rotulos_da_variavel(c).index(r)) for c, r in CATEGORIAS_ANALISADAS]
    tabela.insert(0, "n_treino", [int(p.sum()) for p in pacientes])
    tabela.insert(1, "n_com_recorrencia", [int((p & (divisao.y_treino == 1)).sum()) for p in pacientes])
    return tabela


def acuracia_por_suavizacao(modelo_nb, X: pd.DataFrame, y: np.ndarray) -> dict[float, float]:
    validacao = validacao_repetida(5)
    return {alpha: float(cross_validate(clone(modelo_nb).set_params(modelo__alpha=alpha), X.values, y,
                                        cv=validacao, scoring="accuracy")["test_score"].mean())
            for alpha in [0.1, 0.5, 1.0, 2.0]}


def intervalo_confianca_razoes(naive_bayes, divisao: DivisaoTreinoTeste, gerador: np.random.RandomState,
                               razoes: pd.Series, reamostragens: int = 500) -> pd.DataFrame:
    """Intervalo de 95% das razões (alpha = 1) por bootstrap do treino."""
    amostras = []
    for _ in range(reamostragens):
        sorteio = gerador.randint(0, len(divisao.X_treino), len(divisao.X_treino))
        modelo = clone(naive_bayes).fit(divisao.X_treino.values[sorteio], divisao.y_treino[sorteio])
        amostras.append(razoes_das_categorias(modelo))
    amostras = pd.DataFrame(amostras)
    return pd.DataFrame({"razao": razoes, "ic_2.5": amostras.quantile(.025), "ic_97.5": amostras.quantile(.975)})


def variacao_entre_divisoes(modelos: dict, X: pd.DataFrame, y: np.ndarray) -> dict[str, pd.DataFrame]:
    """Acurácia, recall e AUC em 50 divisões treino/teste (5 partições x 10 repetições)."""
    validacao = validacao_repetida(10)
    resultados = {}
    for nome, modelo in modelos.items():
        cv = cross_validate(clone(modelo), X.values, y, cv=validacao, scoring=["accuracy", "recall", "roc_auc"])
        resultados[nome] = pd.DataFrame({"acuracia": cv["test_accuracy"], "recall": cv["test_recall"],
                                         "auc": cv["test_roc_auc"]})
    return resultados


def resumir_variacao(resultados: dict) -> pd.DataFrame:
    return pd.concat({nome: tabela.describe(percentiles=[.025, .975]).T[["mean", "2.5%", "97.5%"]]
                      for nome, tabela in resultados.items()})


def desempenho_por_subgrupo(probabilidades: dict, dataset: pd.DataFrame, y: np.ndarray, nomes_modelos) -> list[dict]:
    grupos = {
        "mulheres": dataset["sexo"].values == "feminino",
        "homens": dataset["sexo"].values == "masculino",
        "idade < 55": dataset["idade"].values < 55,
        "idade >= 55": dataset["idade"].values >= 55,
    }
    linhas = []
    for grupo, pertence in grupos.items():
        for nome in nomes_modelos:
            prob, y_grupo = probabilidades[nome][pertence], y[pertence]
            linhas.append({"grupo": grupo, "n": int(pertence.sum()), "prop_recorrencia": y_grupo.mean(),
                           "modelo": nome, "acuracia": accuracy_score(y_grupo, classificar(prob)),
                           "recall": recall_score(y_grupo, classificar(prob))})
    return linhas


def estabilidade_da_arvore(legivel: ArvoreLegivel, hiperparametros: dict, divisao: DivisaoTreinoTeste,
                           gerador: np.random.RandomState, reamostragens: int = 200) -> dict:
    """Com que frequência cada variável aparece na raiz e quais conjuntos de variáveis a árvore usa."""
    X_treino_onehot = legivel.transformar(divisao.X_treino)
    raizes, conjuntos = [], []
    for _ in range(reamostragens):
        sorteio = gerador.randint(0, len(divisao.X_treino), len(divisao.X_treino))
        arvore = DecisionTreeClassifier(random_state=SEMENTE, **hiperparametros[ARVORE])
        estrutura = arvore.fit(X_treino_onehot[sorteio], divisao.y_treino[sorteio]).tree_
        usadas = sorted({variavel_de_origem(legivel.colunas_onehot[f]) for f in estrutura.feature if f >= 0})
        raizes.append(variavel_de_origem(legivel.colunas_onehot[estrutura.feature[0]]))
        conjuntos.append(" + ".join(usadas))
    return {"raiz": pd.Series(raizes).value_counts(normalize=True).round(3).to_dict(),
            "variaveis_usadas": pd.Series(conjuntos).value_counts(normalize=True).round(3).head(5).to_dict()}


def comparar_proporcoes_de_divisao(X: pd.DataFrame, y: np.ndarray, sementes: int = 30) -> pd.DataFrame:
    """Repete divisão, busca de hiperparâmetros e teste para 80/20 e 70/30 com várias sementes."""
    linhas = []
    for proporcao_teste in [0.2, 0.3]:
        for semente in range(sementes):
            X_treino, X_teste, y_treino, y_teste = train_test_split(
                X.values, y, test_size=proporcao_teste, stratify=y, random_state=semente)
            for nome, modelo in criar_modelos(ESQUEMA_PRINCIPAL).items():
                busca = buscar_hiperparametros(modelo, nome, X_treino, y_treino, criar_validacao_cruzada(semente))
                prob = probabilidade_de_recorrencia(busca.best_estimator_, X_teste)
                linhas.append({
                    "divisao": f"{round((1 - proporcao_teste) * 100)}/{round(proporcao_teste * 100)}",
                    "semente": semente, "modelo": nome,
                    "acc_teste": accuracy_score(y_teste, classificar(prob)),
                    "recall": recall_score(y_teste, classificar(prob)),
                    "f1": f1_score(y_teste, classificar(prob)),
                    "auc": roc_auc_score(y_teste, prob),
                    "acc_treino": busca.best_estimator_.score(X_treino, y_treino),
                })
    return pd.DataFrame(linhas)


def resumir_proporcoes_de_divisao(tabela: pd.DataFrame) -> pd.DataFrame:
    return tabela.groupby(["modelo", "divisao"]).agg(
        acc=("acc_teste", "mean"), acc_dp=("acc_teste", "std"), recall=("recall", "mean"),
        f1=("f1", "mean"), auc=("auc", "mean")).round(3)
