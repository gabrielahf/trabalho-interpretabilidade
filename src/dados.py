"""Carregamento, tradução e codificação do dataset, e divisão em treino e teste."""

import io
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (ARQUIVO_DATASET, CATEGORIAS, COLUNA_ALVO, NOMES_COLUNAS,
                        PASTA_DADOS, PROPORCAO_TESTE, SEMENTE, URL_DATASET,
                        VARIAVEIS_CATEGORICAS, VARIAVEIS_NUMERICAS)


@dataclass
class EsquemaVariaveis:
    """Variáveis usadas no modelo. Em X, as numéricas vêm primeiro e as categóricas depois."""

    numericas: list[str]
    categoricas: list[str]

    @property
    def variaveis(self) -> list[str]:
        return self.numericas + self.categoricas

    @property
    def indices_numericas(self) -> list[int]:
        return list(range(len(self.numericas)))

    @property
    def indices_categoricas(self) -> list[int]:
        inicio = len(self.numericas)
        return list(range(inicio, inicio + len(self.categoricas)))

    @property
    def quantidade_categorias(self) -> list[int]:
        return [len(CATEGORIAS[coluna]) for coluna in self.categoricas]


ESQUEMA_PRINCIPAL = EsquemaVariaveis(VARIAVEIS_NUMERICAS, VARIAVEIS_CATEGORICAS)


@dataclass
class DivisaoTreinoTeste:
    X_treino: pd.DataFrame
    X_teste: pd.DataFrame
    y_treino: np.ndarray
    y_teste: np.ndarray


def rotulos_da_variavel(coluna: str) -> list[str]:
    """Rótulos de uma variável categórica, na ordem do código numérico."""
    return list(CATEGORIAS[coluna].values())


def baixar_dataset_se_necessario() -> None:
    if ARQUIVO_DATASET.exists():
        return
    print("Baixando o dataset da UCI...")
    PASTA_DADOS.mkdir(exist_ok=True)
    conteudo_zip = urllib.request.urlopen(URL_DATASET).read()
    zipfile.ZipFile(io.BytesIO(conteudo_zip)).extractall(PASTA_DADOS)


def carregar_dataset_original() -> pd.DataFrame:
    """Lê o CSV e renomeia as colunas para português."""
    baixar_dataset_se_necessario()
    return pd.read_csv(ARQUIVO_DATASET).rename(columns=NOMES_COLUNAS)


def traduzir_dataset(dataset_original: pd.DataFrame) -> pd.DataFrame:
    """Traduz as categorias e transforma o alvo em 0 (não recorre) e 1 (recorre)."""
    dataset = dataset_original.copy()
    for coluna, traducao in CATEGORIAS.items():
        dataset[coluna] = dataset[coluna].map(traducao)
    if dataset.isna().any().any():
        raise ValueError("Há categorias no dataset que não estão em CATEGORIAS.")
    dataset[COLUNA_ALVO] = (dataset[COLUNA_ALVO] == "Yes").astype(int)
    return dataset


def codificar_variaveis(dataset: pd.DataFrame, esquema: EsquemaVariaveis = ESQUEMA_PRINCIPAL) -> pd.DataFrame:
    """Monta a matriz X, trocando cada categoria pelo seu código inteiro."""
    X = dataset[esquema.variaveis].copy()
    for coluna in esquema.categoricas:
        codigo_da_categoria = {rotulo: codigo for codigo, rotulo in enumerate(rotulos_da_variavel(coluna))}
        X[coluna] = X[coluna].map(codigo_da_categoria).astype(int)
    return X


def dividir_treino_teste(X: pd.DataFrame, y: np.ndarray) -> DivisaoTreinoTeste:
    """Divisão 80/20 estratificada pela classe."""
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=PROPORCAO_TESTE, stratify=y, random_state=SEMENTE)
    return DivisaoTreinoTeste(X_treino.reset_index(drop=True), X_teste.reset_index(drop=True),
                              y_treino, y_teste)


def descrever_paciente(X: pd.DataFrame, posicao: int, esquema: EsquemaVariaveis = ESQUEMA_PRINCIPAL) -> dict:
    """Converte uma linha codificada de X de volta para rótulos legíveis."""
    linha = X.iloc[posicao]
    descricao = {}
    for coluna in esquema.variaveis:
        if coluna in esquema.categoricas:
            descricao[coluna] = rotulos_da_variavel(coluna)[int(linha[coluna])]
        else:
            descricao[coluna] = int(linha[coluna])
    return descricao


def resumir_dataset(dataset_original: pd.DataFrame, dataset: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "instancias": len(dataset_original),
        "linhas_identicas": int(dataset_original.duplicated().sum()),
        "prop_recorrencia": round(dataset[COLUNA_ALVO].mean(), 3),
    })
