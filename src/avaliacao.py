import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate, learning_curve, validation_curve

from src.config import ARVORE, KNN, LIMIAR_DECISAO, SEMENTE
from src.dados import DivisaoTreinoTeste
from src.modelos import com_prefixo, criar_validacao_cruzada


def probabilidade_de_recorrencia(modelo, X) -> np.ndarray:
    return modelo.predict_proba(np.asarray(X, dtype=float))[:, 1]


def classificar(probabilidades: np.ndarray) -> np.ndarray:
    return (probabilidades >= LIMIAR_DECISAO).astype(int)


def calcular_metricas(y_real, probabilidades) -> dict:
    y_previsto = classificar(probabilidades)
    return {
        "acuracia": accuracy_score(y_real, y_previsto),
        "acuracia_balanceada": balanced_accuracy_score(y_real, y_previsto),
        "precisao": precision_score(y_real, y_previsto),
        "recall": recall_score(y_real, y_previsto),
        "especificidade": recall_score(y_real, y_previsto, pos_label=0),
        "f1": f1_score(y_real, y_previsto),
        "roc_auc": roc_auc_score(y_real, probabilidades),
    }


def avaliar_no_teste(modelos: dict, divisao: DivisaoTreinoTeste) -> pd.DataFrame:
    linhas = []
    for nome, modelo in modelos.items():
        probabilidades = probabilidade_de_recorrencia(modelo, divisao.X_teste)
        linhas.append({"modelo": nome, **calcular_metricas(divisao.y_teste, probabilidades)})
    return pd.DataFrame(linhas).set_index("modelo").round(3)


def comparar_treino_validacao_teste(modelos: dict, divisao: DivisaoTreinoTeste) -> pd.DataFrame:
    """Diferenças pequenas entre treino, validação cruzada e teste indicam ausência de overfitting."""
    validacao = criar_validacao_cruzada()
    X_treino = divisao.X_treino.values
    linhas = []
    for nome, modelo in modelos.items():
        prob_treino = probabilidade_de_recorrencia(modelo, X_treino)
        prob_teste = probabilidade_de_recorrencia(modelo, divisao.X_teste)
        cv = cross_validate(modelo, X_treino, divisao.y_treino, cv=validacao, scoring=["accuracy", "f1", "roc_auc"])
        linhas.append({
            "modelo": nome,
            "acc_treino": accuracy_score(divisao.y_treino, classificar(prob_treino)),
            "acc_cv": cv["test_accuracy"].mean(),
            "acc_cv_dp": cv["test_accuracy"].std(),
            "acc_teste": accuracy_score(divisao.y_teste, classificar(prob_teste)),
            "f1_treino": f1_score(divisao.y_treino, classificar(prob_treino)),
            "f1_cv": cv["test_f1"].mean(),
            "f1_teste": f1_score(divisao.y_teste, classificar(prob_teste)),
            "auc_treino": roc_auc_score(divisao.y_treino, prob_treino),
            "auc_cv": cv["test_roc_auc"].mean(),
            "auc_teste": roc_auc_score(divisao.y_teste, prob_teste),
        })
    return pd.DataFrame(linhas).set_index("modelo").round(3)


def acuracia_de_referencia(y_teste: np.ndarray) -> float:
    """Acurácia de quem sempre chuta a classe majoritária."""
    return float(max(y_teste.mean(), 1 - y_teste.mean()))


def curva_profundidade_arvore(modelos_base: dict, hiperparametros: dict, divisao: DivisaoTreinoTeste) -> dict:
    """F1 no treino e na validação para cada profundidade, sem limite de folha mínima."""
    profundidades = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    arvore = clone(modelos_base[ARVORE]).set_params(
        modelo__criterion=hiperparametros[ARVORE]["criterion"], modelo__min_samples_leaf=1)
    f1_treino, f1_validacao = validation_curve(
        arvore, divisao.X_treino.values, divisao.y_treino, param_name="modelo__max_depth",
        param_range=profundidades, cv=criar_validacao_cruzada(), scoring="f1")
    return {"valores": profundidades, "treino": f1_treino, "validacao": f1_validacao,
            "escolhido": hiperparametros[ARVORE]["max_depth"]}


def curva_vizinhos_knn(modelos_base: dict, hiperparametros: dict, divisao: DivisaoTreinoTeste) -> dict:
    """F1 no treino e na validação para cada número de vizinhos."""
    vizinhos = list(range(1, 80, 4))
    knn = clone(modelos_base[KNN]).set_params(modelo__metric=hiperparametros[KNN]["metric"])
    f1_treino, f1_validacao = validation_curve(
        knn, divisao.X_treino.values, divisao.y_treino, param_name="modelo__n_neighbors",
        param_range=vizinhos, cv=criar_validacao_cruzada(), scoring="f1")
    return {"valores": vizinhos, "treino": f1_treino, "validacao": f1_validacao,
            "escolhido": hiperparametros[KNN]["n_neighbors"]}


def avaliar_arvore_sem_limite(modelos_base: dict, divisao: DivisaoTreinoTeste) -> dict:
    """F1 de uma árvore sem limite de profundidade, para mostrar o overfitting."""
    arvore = clone(modelos_base[ARVORE]).set_params(modelo__max_depth=None)
    cv = cross_validate(arvore, divisao.X_treino.values, divisao.y_treino, cv=criar_validacao_cruzada(),
                        scoring="f1", return_train_score=True)
    return {"treino": float(cv["train_score"].mean()), "cv": float(cv["test_score"].mean())}


def curvas_de_aprendizado(modelos: dict, divisao: DivisaoTreinoTeste) -> dict:
    curvas = {}
    for nome, modelo in modelos.items():
        tamanhos, f1_treino, f1_validacao = learning_curve(
            modelo, divisao.X_treino.values, divisao.y_treino, cv=criar_validacao_cruzada(), scoring="f1",
            train_sizes=np.linspace(.2, 1, 8), random_state=SEMENTE)
        curvas[nome] = {"tamanhos": tamanhos, "treino": f1_treino, "validacao": f1_validacao}
    return curvas


def configurar_modelo(modelo_base, hiperparametros: dict):
    """Cópia de um pipeline com os hiperparâmetros já escolhidos."""
    return clone(modelo_base).set_params(**com_prefixo(hiperparametros))
