"""Funções para salvar figuras, tabelas e arquivos JSON, e para imprimir títulos no terminal."""

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PASTA_FIGURAS, PASTA_RESULTADOS


def criar_pastas_de_saida() -> None:
    PASTA_FIGURAS.mkdir(exist_ok=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)


def salvar_figura(nome_arquivo: str) -> None:
    """Salva a figura atual do matplotlib em figuras/ e fecha a figura."""
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / nome_arquivo, dpi=150, bbox_inches="tight")
    plt.close()


def salvar_tabela(tabela: pd.DataFrame | pd.Series, nome_arquivo: str, **opcoes) -> None:
    """Salva um DataFrame ou Series em resultados/ como CSV."""
    tabela.to_csv(PASTA_RESULTADOS / nome_arquivo, **opcoes)


def salvar_texto(texto: str, nome_arquivo: str) -> None:
    (PASTA_RESULTADOS / nome_arquivo).write_text(texto, encoding="utf-8")


def salvar_json(conteudo: dict, nome_arquivo: str) -> None:
    with open(PASTA_RESULTADOS / nome_arquivo, "w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, indent=2, ensure_ascii=False, default=float)


def imprimir_titulo(titulo: str) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")
