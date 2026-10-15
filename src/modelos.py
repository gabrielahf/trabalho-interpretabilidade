"""Criação dos modelos (Árvore, Naive Bayes misto e KNN) e escolha dos hiperparâmetros."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (ARVORE, GRADES_HIPERPARAMETROS, KNN, NAIVE_BAYES,
                        NUMERO_PARTICOES_CV, SEMENTE)
from src.dados import EsquemaVariaveis


class NaiveBayesMisto(ClassifierMixin, BaseEstimator):
    """Naive Bayes com distribuição Normal nas variáveis numéricas e categórica nas demais.

    P(classe | x) é proporcional a P(classe) * produto de P(x_j | classe).
    Em escala log, a decisão é uma soma, então a contribuição de cada variável pode ser lida diretamente.
    """

    def __init__(self, alpha=1.0, indices_numericas=None, indices_categoricas=None, quantidade_categorias=None):
        self.alpha = alpha
        self.indices_numericas = indices_numericas
        self.indices_categoricas = indices_categoricas
        self.quantidade_categorias = quantidade_categorias

    def _codigos_categoricos(self, X):
        X = np.asarray(X, dtype=float)
        codigos = np.rint(X[:, self.indices_categoricas]).astype(int)
        return np.clip(codigos, 0, np.array(self.quantidade_categorias) - 1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        X_numerico = X[:, self.indices_numericas]
        self.media_ = np.array([X_numerico[y == classe].mean(axis=0) for classe in self.classes_])
        self.desvio_ = np.array([X_numerico[y == classe].std(axis=0) for classe in self.classes_]) + 1e-6
        self.priori_ = np.array([(y == classe).mean() for classe in self.classes_])
        self.nb_categorico_ = CategoricalNB(alpha=self.alpha, min_categories=self.quantidade_categorias)
        self.nb_categorico_.fit(self._codigos_categoricos(X), y)
        return self

    def log_verossimilhanca_por_variavel(self, X):
        """log P(x_j | classe) de cada paciente, variável e classe. Formato: (pacientes, variáveis, classes)."""
        X = np.asarray(X, dtype=float)
        numero_variaveis = len(self.indices_numericas) + len(self.indices_categoricas)
        resultado = np.zeros((X.shape[0], numero_variaveis, len(self.classes_)))
        for classe in range(len(self.classes_)):
            resultado[:, self.indices_numericas, classe] = norm.logpdf(
                X[:, self.indices_numericas], self.media_[classe], self.desvio_[classe])
        codigos = self._codigos_categoricos(X)
        for posicao, indice_coluna in enumerate(self.indices_categoricas):
            log_prob = self.nb_categorico_.feature_log_prob_[posicao]
            resultado[:, indice_coluna, :] = log_prob[:, codigos[:, posicao]].T
        return resultado

    def razao_verossimilhanca(self, posicao_categorica: int, codigo: int) -> float:
        """P(categoria | recorre) / P(categoria | não recorre) para uma categoria."""
        log_prob = self.nb_categorico_.feature_log_prob_[posicao_categorica][:, codigo]
        return float(np.exp(log_prob[1] - log_prob[0]))

    def predict_proba(self, X):
        log_posterior = np.log(self.priori_) + self.log_verossimilhanca_por_variavel(X).sum(axis=1)
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        probabilidades = np.exp(log_posterior)
        return probabilidades / probabilidades.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def criar_preprocessador_onehot(esquema: EsquemaVariaveis) -> ColumnTransformer:
    """Padroniza as numéricas e aplica one-hot nas categóricas (usado na Árvore e no KNN)."""
    categorias_possiveis = [list(range(quantidade)) for quantidade in esquema.quantidade_categorias]
    return ColumnTransformer([
        ("numericas", StandardScaler(), esquema.indices_numericas),
        ("categoricas", OneHotEncoder(categories=categorias_possiveis, handle_unknown="ignore"),
         esquema.indices_categoricas),
    ])


def criar_modelos(esquema: EsquemaVariaveis) -> dict[str, Pipeline]:
    """Cria os três pipelines ainda sem treinar."""
    naive_bayes = NaiveBayesMisto(indices_numericas=esquema.indices_numericas,
                                  indices_categoricas=esquema.indices_categoricas,
                                  quantidade_categorias=esquema.quantidade_categorias)
    return {
        ARVORE: Pipeline([("preprocessamento", criar_preprocessador_onehot(esquema)),
                          ("modelo", DecisionTreeClassifier(random_state=SEMENTE))]),
        NAIVE_BAYES: Pipeline([("modelo", naive_bayes)]),
        KNN: Pipeline([("preprocessamento", criar_preprocessador_onehot(esquema)),
                       ("modelo", KNeighborsClassifier())]),
    }


def criar_validacao_cruzada(semente: int = SEMENTE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=NUMERO_PARTICOES_CV, shuffle=True, random_state=semente)


def sem_prefixo(parametros: dict) -> dict:
    """Remove o prefixo 'modelo__' dos nomes dos parâmetros."""
    return {nome.replace("modelo__", ""): valor for nome, valor in parametros.items()}


def com_prefixo(parametros: dict) -> dict:
    return {f"modelo__{nome}": valor for nome, valor in parametros.items()}


def buscar_hiperparametros(modelo: Pipeline, nome_modelo: str, X, y, validacao) -> GridSearchCV:
    """Busca em grade usando o F1 da classe 'recorre' como critério."""
    busca = GridSearchCV(modelo, GRADES_HIPERPARAMETROS[nome_modelo], cv=validacao, scoring="f1", n_jobs=-1)
    return busca.fit(X, y)


def escolher_arvore_mais_simples(busca: GridSearchCV) -> dict:
    """Entre as configurações a até 1 desvio-padrão da melhor, escolhe a árvore mais rasa."""
    resultados = pd.DataFrame(busca.cv_results_)
    melhor = resultados[resultados["rank_test_score"] == 1].iloc[0]
    limite = resultados["mean_test_score"].max() - melhor["std_test_score"]
    candidatas = resultados[resultados["mean_test_score"] >= limite].sort_values(
        ["param_modelo__max_depth", "param_modelo__min_samples_leaf", "mean_test_score"],
        ascending=[True, False, False])
    return candidatas.iloc[0]["params"]


def treinar_modelos(esquema: EsquemaVariaveis, X_treino, y_treino) -> tuple[dict, dict, dict]:
    """Escolhe os hiperparâmetros por validação cruzada e devolve os modelos treinados."""
    validacao = criar_validacao_cruzada()
    modelos_base = criar_modelos(esquema)
    modelos_treinados, hiperparametros = {}, {}
    for nome, modelo in modelos_base.items():
        busca = buscar_hiperparametros(modelo, nome, X_treino, y_treino, validacao)
        modelos_treinados[nome] = busca.best_estimator_
        hiperparametros[nome] = sem_prefixo(busca.best_params_)
        print(f"{nome}: {hiperparametros[nome]} | F1 na validação = {busca.best_score_:.3f}")
        if nome == ARVORE:
            parametros_arvore = escolher_arvore_mais_simples(busca)
            modelos_treinados[nome] = clone(modelo).set_params(**parametros_arvore).fit(X_treino, y_treino)
            hiperparametros[nome] = sem_prefixo(parametros_arvore)
            print(f"{nome} (mais simples a até 1 desvio): {hiperparametros[nome]}")
    return modelos_treinados, hiperparametros, modelos_base
