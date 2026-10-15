from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

from src.avaliacao import classificar, probabilidade_de_recorrencia
from src.config import ARVORE, KNN, NAIVE_BAYES, ROTULOS_CLASSES, SEMENTE
from src.dados import ESQUEMA_PRINCIPAL, DivisaoTreinoTeste, rotulos_da_variavel

VARIAVEIS = ESQUEMA_PRINCIPAL.variaveis
CATEGORICAS = ESQUEMA_PRINCIPAL.categoricas


@dataclass
class ArvoreLegivel:
    """Cópia da árvore treinada com a idade em anos, para que as regras fiquem legíveis."""

    arvore: DecisionTreeClassifier
    colunas_onehot: list[str]
    codificador_categorico: object

    def transformar(self, X: pd.DataFrame) -> np.ndarray:
        categoricas = self.codificador_categorico.transform(X.values[:, ESQUEMA_PRINCIPAL.indices_categoricas])
        return np.hstack([X[ESQUEMA_PRINCIPAL.numericas].values, categoricas.toarray()])


def variavel_de_origem(coluna_onehot: str) -> str:
    """'risco=alto' -> 'risco'."""
    return coluna_onehot.split("=")[0]


def criar_arvore_legivel(modelo_arvore, hiperparametros: dict, divisao: DivisaoTreinoTeste) -> ArvoreLegivel:
    """Retreina a árvore sem padronizar a idade. Como a árvore não depende de escala, as decisões são as mesmas."""
    colunas_onehot = ESQUEMA_PRINCIPAL.numericas + [
        f"{coluna}={rotulo}" for coluna in CATEGORICAS for rotulo in rotulos_da_variavel(coluna)]
    codificador = modelo_arvore.named_steps["preprocessamento"].named_transformers_["categoricas"]
    legivel = ArvoreLegivel(None, colunas_onehot, codificador)
    X_treino_onehot = legivel.transformar(divisao.X_treino)
    legivel.arvore = DecisionTreeClassifier(random_state=SEMENTE, **hiperparametros).fit(X_treino_onehot, divisao.y_treino)
    previsoes_originais = modelo_arvore.predict(divisao.X_treino.values)
    if not np.array_equal(legivel.arvore.predict(X_treino_onehot), previsoes_originais):
        raise RuntimeError("A árvore legível não reproduz a árvore treinada.")
    return legivel


def regras_da_arvore(legivel: ArvoreLegivel) -> str:
    return export_text(legivel.arvore, feature_names=legivel.colunas_onehot, decimals=1, show_weights=True)


def importancia_gini(legivel: ArvoreLegivel) -> pd.Series:
    """Importância de Gini somada por variável original (soma as colunas do one-hot)."""
    por_coluna = pd.Series(legivel.arvore.feature_importances_, index=legivel.colunas_onehot)
    por_variavel = por_coluna.groupby(variavel_de_origem).sum().reindex(VARIAVEIS)
    return por_variavel.sort_values(ascending=False)


def caminho_na_arvore(legivel: ArvoreLegivel, paciente: pd.DataFrame) -> list[str]:
    """Perguntas que a árvore fez para o paciente, da raiz até a folha."""
    paciente_onehot = legivel.transformar(paciente)
    estrutura = legivel.arvore.tree_
    caminho = []
    for no in legivel.arvore.decision_path(paciente_onehot).indices:
        if estrutura.children_left[no] == -1:
            proporcoes = estrutura.value[no][0] / estrutura.value[no][0].sum()
            caminho.append(f"FOLHA: {proporcoes.round(3).tolist()} (proporção sem/com)")
        else:
            coluna = legivel.colunas_onehot[estrutura.feature[no]]
            corte = estrutura.threshold[no]
            valor = paciente_onehot[0, estrutura.feature[no]]
            caminho.append(f"{coluna} = {valor:.1f} {'<=' if valor <= corte else '>'} {corte:.1f}")
    return caminho


def probabilidades_condicionais(naive_bayes, X_treino: pd.DataFrame) -> pd.DataFrame:
    """P(categoria | classe) e razão de verossimilhança de todas as categorias."""
    linhas = []
    for posicao, coluna in enumerate(CATEGORICAS):
        probabilidades = np.exp(naive_bayes.nb_categorico_.feature_log_prob_[posicao])
        for codigo, rotulo in enumerate(rotulos_da_variavel(coluna)):
            linhas.append({
                "variavel": coluna,
                "categoria": rotulo,
                "n_treino": int((X_treino[coluna] == codigo).sum()),
                "P(x|sem recorrência)": probabilidades[0, codigo],
                "P(x|com recorrência)": probabilidades[1, codigo],
                "razao_verossimilhanca": probabilidades[1, codigo] / probabilidades[0, codigo],
            })
    return pd.DataFrame(linhas).round(3)


def distribuicao_da_idade(naive_bayes) -> pd.DataFrame:
    return pd.DataFrame({
        "media_sem": naive_bayes.media_[0], "dp_sem": naive_bayes.desvio_[0],
        "media_com": naive_bayes.media_[1], "dp_com": naive_bayes.desvio_[1],
    }, index=ESQUEMA_PRINCIPAL.numericas).round(2)


def log_razao_por_variavel(naive_bayes, X) -> np.ndarray:
    """log[P(x_j | recorre) / P(x_j | não recorre)] de cada paciente e variável."""
    log_verossimilhanca = naive_bayes.log_verossimilhanca_por_variavel(np.asarray(X, dtype=float))
    return log_verossimilhanca[:, :, 1] - log_verossimilhanca[:, :, 0]


def importancia_naive_bayes(naive_bayes, X_teste) -> pd.Series:
    """Média do |log da razão| de cada variável nos pacientes de teste."""
    media = np.abs(log_razao_por_variavel(naive_bayes, X_teste)).mean(axis=0)
    return pd.Series(media, index=VARIAVEIS).sort_values(ascending=False)


def importancia_por_permutacao(modelos: dict, divisao: DivisaoTreinoTeste) -> dict[str, pd.DataFrame]:
    """Queda média da AUC no teste ao embaralhar cada variável (30 repetições)."""
    importancias = {}
    for nome, modelo in modelos.items():
        resultado = permutation_importance(modelo, divisao.X_teste.values, divisao.y_teste,
                                           scoring="roc_auc", n_repeats=30, random_state=SEMENTE)
        importancias[nome] = pd.DataFrame({"media": resultado.importances_mean, "dp": resultado.importances_std},
                                          index=VARIAVEIS)
    return importancias


def tabela_permutacao(importancias: dict) -> pd.DataFrame:
    return pd.concat({nome: tabela["media"] for nome, tabela in importancias.items()}, axis=1).round(4)


def calcular_shap(modelos: dict, divisao: DivisaoTreinoTeste) -> dict[str, shap.Explanation]:
    """SHAP (KernelExplainer) da probabilidade de recorrência, com 60 pacientes do treino como referência."""
    referencia = shap.sample(divisao.X_treino, 60, random_state=SEMENTE)
    explicacoes = {}
    for nome, modelo in modelos.items():
        explicador = shap.KernelExplainer(lambda X, m=modelo: probabilidade_de_recorrencia(m, X), referencia.values)
        valores = explicador.shap_values(divisao.X_teste.values, nsamples=500, silent=True)
        explicacoes[nome] = shap.Explanation(
            values=valores, base_values=np.full(len(divisao.X_teste), explicador.expected_value),
            data=divisao.X_teste.values, feature_names=VARIAVEIS)
        print(f"SHAP calculado: {nome}")
    return explicacoes


def importancia_shap(explicacoes: dict) -> pd.DataFrame:
    return pd.DataFrame({nome: np.abs(explicacao.values).mean(axis=0) for nome, explicacao in explicacoes.items()},
                        index=VARIAVEIS)


def shap_do_paciente(explicacoes: dict, posicao: int) -> pd.DataFrame:
    return pd.DataFrame({nome: explicacao.values[posicao] for nome, explicacao in explicacoes.items()},
                        index=VARIAVEIS).round(3)


def probabilidades_no_teste(modelos: dict, divisao: DivisaoTreinoTeste) -> pd.DataFrame:
    return pd.DataFrame({nome: probabilidade_de_recorrencia(modelo, divisao.X_teste) for nome, modelo in modelos.items()})


def escolher_caso_exemplo(probabilidades: pd.DataFrame, y_teste: np.ndarray) -> int:
    """Paciente com recorrência que os três modelos acertam, mas com as confianças mais diferentes."""
    previsoes = classificar(probabilidades.values)
    todos_acertam = (previsoes == y_teste[:, None]).all(axis=1) & (y_teste == 1)
    return int(probabilidades[todos_acertam].std(axis=1).idxmax())


def concordancia_entre_modelos(probabilidades: pd.DataFrame) -> tuple[dict, int]:
    """Proporção de pacientes em que cada par de modelos (e os três) dão a mesma resposta."""
    previsoes = pd.DataFrame(classificar(probabilidades.values), columns=probabilidades.columns)
    nomes = list(previsoes.columns)
    concordancia = {f"{a} x {b}": float((previsoes[a] == previsoes[b]).mean())
                    for i, a in enumerate(nomes) for b in nomes[i + 1:]}
    mesma_resposta = previsoes.nunique(axis=1) == 1
    concordancia["os três"] = float(mesma_resposta.mean())
    return concordancia, int((~mesma_resposta).sum())


def vizinhos_do_paciente(modelo_knn, paciente: pd.DataFrame, divisao: DivisaoTreinoTeste) -> pd.DataFrame:
    """Os k vizinhos mais próximos do paciente no treino, com rótulos legíveis."""
    paciente_transformado = modelo_knn.named_steps["preprocessamento"].transform(paciente.values)
    distancias, posicoes = modelo_knn.named_steps["modelo"].kneighbors(paciente_transformado)
    vizinhos = divisao.X_treino.iloc[posicoes[0]].copy()
    for coluna in CATEGORICAS:
        rotulos = rotulos_da_variavel(coluna)
        vizinhos[coluna] = vizinhos[coluna].map(lambda codigo, r=rotulos: r[int(codigo)])
    return vizinhos.assign(recorrencia=divisao.y_treino[posicoes[0]], distancia=distancias[0].round(2))


def criar_explicador_lime(X_treino: pd.DataFrame, semente: int) -> LimeTabularExplainer:
    nomes_categorias = {indice: rotulos_da_variavel(coluna)
                        for indice, coluna in zip(ESQUEMA_PRINCIPAL.indices_categoricas, CATEGORICAS)}
    return LimeTabularExplainer(X_treino.values.astype(float), feature_names=VARIAVEIS,
                                class_names=ROTULOS_CLASSES,
                                categorical_features=ESQUEMA_PRINCIPAL.indices_categoricas,
                                categorical_names=nomes_categorias, discretize_continuous=True,
                                random_state=semente)


def explicar_com_lime(modelos: dict, paciente: pd.DataFrame, X_treino: pd.DataFrame) -> dict:
    """Explicação LIME do paciente em cada modelo (8 variáveis, 5000 amostras)."""
    explicador = criar_explicador_lime(X_treino, SEMENTE)
    return {nome: explicador.explain_instance(paciente.values[0].astype(float), modelo.predict_proba,
                                              num_features=8, num_samples=5000)
            for nome, modelo in modelos.items()}


def estabilidade_do_lime(modelo, paciente: pd.DataFrame, X_treino: pd.DataFrame, repeticoes: int = 10) -> float:
    """Fração das execuções (sementes diferentes) que repetem as mesmas 3 variáveis principais."""
    principais = []
    for semente in range(repeticoes):
        explicacao = criar_explicador_lime(X_treino, semente).explain_instance(
            paciente.values[0].astype(float), modelo.predict_proba, num_features=3, num_samples=1000)
        principais.append(tuple(sorted(indice for indice, _ in explicacao.as_map()[1])))
    return float(pd.Series(principais).value_counts().iloc[0] / repeticoes)


def contribuicoes_naive_bayes_paciente(naive_bayes, paciente: pd.DataFrame) -> tuple[pd.Series, float]:
    """Contribuição exata de cada variável no log da chance do paciente, e o log da chance a priori."""
    contribuicoes = pd.Series(log_razao_por_variavel(naive_bayes, paciente.values)[0], index=VARIAVEIS)
    log_chance_priori = float(np.log(naive_bayes.priori_[1] / naive_bayes.priori_[0]))
    return contribuicoes, log_chance_priori


def montar_rankings(importancia_arvore, importancia_nb, permutacao: dict, shap_medio: pd.DataFrame) -> pd.DataFrame:
    """Reúne as oito medidas de importância em uma tabela (variáveis nas linhas)."""
    return pd.DataFrame({
        "Árvore (Gini)": importancia_arvore,
        "NB (|log-razão| média)": importancia_nb,
        "Árvore (permutação)": permutacao[ARVORE]["media"].round(4),
        "NB (permutação)": permutacao[NAIVE_BAYES]["media"].round(4),
        "KNN (permutação)": permutacao[KNN]["media"].round(4),
        "Árvore (SHAP)": shap_medio[ARVORE],
        "NB (SHAP)": shap_medio[NAIVE_BAYES],
        "KNN (SHAP)": shap_medio[KNN],
    }).reindex(VARIAVEIS)


def correlacao_entre_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    return rankings.corr(method="spearman")
