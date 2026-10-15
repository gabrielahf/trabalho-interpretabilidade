"""Testes do NaiveBayesMisto. Uso: python -m unittest discover tests"""

import unittest

import numpy as np

from src.modelos import NaiveBayesMisto


def dados_de_exemplo():
    gerador = np.random.RandomState(0)
    idade = np.concatenate([gerador.normal(40, 10, 50), gerador.normal(60, 10, 50)])
    categoria = np.concatenate([gerador.choice(3, 50, p=[.7, .2, .1]), gerador.choice(3, 50, p=[.1, .2, .7])])
    X = np.column_stack([idade, categoria])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


def criar_modelo():
    return NaiveBayesMisto(indices_numericas=[0], indices_categoricas=[1], quantidade_categorias=[3])


class TestNaiveBayesMisto(unittest.TestCase):

    def test_probabilidades_somam_um(self):
        X, y = dados_de_exemplo()
        probabilidades = criar_modelo().fit(X, y).predict_proba(X)
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0)

    def test_aprende_media_e_priori_por_classe(self):
        X, y = dados_de_exemplo()
        modelo = criar_modelo().fit(X, y)
        np.testing.assert_allclose(modelo.priori_, [0.5, 0.5])
        self.assertAlmostEqual(modelo.media_[0, 0], X[y == 0, 0].mean())
        self.assertAlmostEqual(modelo.media_[1, 0], X[y == 1, 0].mean())

    def test_contribuicoes_reproduzem_a_probabilidade(self):
        """A soma das contribuições por variável deve reproduzir a probabilidade prevista."""
        X, y = dados_de_exemplo()
        modelo = criar_modelo().fit(X, y)
        log_verossimilhanca = modelo.log_verossimilhanca_por_variavel(X)
        log_chance = np.log(modelo.priori_[1] / modelo.priori_[0]) + (
            log_verossimilhanca[:, :, 1] - log_verossimilhanca[:, :, 0]).sum(axis=1)
        np.testing.assert_allclose(1 / (1 + np.exp(-log_chance)), modelo.predict_proba(X)[:, 1])

    def test_razao_de_verossimilhanca(self):
        X, y = dados_de_exemplo()
        modelo = criar_modelo().fit(X, y)
        self.assertLess(modelo.razao_verossimilhanca(0, 0), 1)
        self.assertGreater(modelo.razao_verossimilhanca(0, 2), 1)

    def test_classifica_bem_dados_separaveis(self):
        X, y = dados_de_exemplo()
        acuracia = (criar_modelo().fit(X, y).predict(X) == y).mean()
        self.assertGreater(acuracia, 0.8)


if __name__ == "__main__":
    unittest.main()
