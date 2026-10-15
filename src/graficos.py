import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.tree import plot_tree

from src.avaliacao import probabilidade_de_recorrencia
from src.config import (ARVORE, COLUNA_ALVO, COR_DESTAQUE, COR_NB_CALIBRADO, COR_NEUTRA,
                        CORES_CLASSES, CORES_MODELOS, ROTULOS_CLASSES, SUFIXO_ARQUIVO_MODELO)
from src.dados import ESQUEMA_PRINCIPAL, rotulos_da_variavel
from src.saida import salvar_figura

sns.set_theme(style="whitegrid", context="notebook")


def classe_e_idade(dataset: pd.DataFrame) -> None:
    _, eixos = plt.subplots(1, 2, figsize=(11, 4))
    dataset[COLUNA_ALVO].map({0: ROTULOS_CLASSES[0], 1: ROTULOS_CLASSES[1]}).value_counts().plot.bar(
        ax=eixos[0], color=CORES_CLASSES, rot=0)
    eixos[0].set(title="Distribuição da classe", xlabel="")
    sns.kdeplot(data=dataset, x="idade", hue=COLUNA_ALVO, fill=True, common_norm=False, ax=eixos[1],
                palette=CORES_CLASSES)
    eixos[1].legend([ROTULOS_CLASSES[1], ROTULOS_CLASSES[0]])
    eixos[1].set_title("Idade por classe")
    salvar_figura("01_classe_e_idade.png")


def taxa_de_recorrencia_por_categoria(dataset: pd.DataFrame) -> None:
    _, eixos = plt.subplots(2, 3, figsize=(16, 8))
    for eixo, coluna in zip(eixos.ravel(), ["risco", "N", "adenopatia", "T", "focalidade", "sexo"]):
        taxa = dataset.groupby(coluna)[COLUNA_ALVO].agg(["mean", "size"]).reindex(rotulos_da_variavel(coluna)).dropna()
        eixo.bar(taxa.index, taxa["mean"], color=COR_DESTAQUE)
        for posicao, (media, quantidade) in enumerate(zip(taxa["mean"], taxa["size"])):
            eixo.text(posicao, media + .02, f"n={int(quantidade)}", ha="center", fontsize=9)
        eixo.set(title=coluna, ylim=(0, 1.1), ylabel="taxa de recorrência")
        eixo.tick_params(axis="x", rotation=20)
    salvar_figura("02_taxa_recorrencia_categorias.png")


def _desenhar_curva(eixo, valores, treino, validacao) -> None:
    eixo.plot(valores, treino.mean(1), "o-", label="treino")
    eixo.plot(valores, validacao.mean(1), "o-", label="validação (CV)")
    eixo.fill_between(valores, validacao.mean(1) - validacao.std(1), validacao.mean(1) + validacao.std(1), alpha=.2)


def curvas_de_validacao(curva_arvore: dict, curva_knn: dict) -> None:
    _, eixos = plt.subplots(1, 2, figsize=(12, 4))
    for eixo, curva, rotulo_x, titulo, legenda in [
            (eixos[0], curva_arvore, "profundidade máxima", "Árvore: profundidade", "lower right"),
            (eixos[1], curva_knn, "k (vizinhos)", "KNN: número de vizinhos", "upper right")]:
        _desenhar_curva(eixo, curva["valores"], curva["treino"], curva["validacao"])
        eixo.axvline(curva["escolhido"], ls="--", c="gray")
        eixo.set(xlabel=rotulo_x, ylabel="F1", title=titulo)
        eixo.legend(loc=legenda)
    salvar_figura("04_curvas_validacao.png")


def curvas_de_aprendizado(curvas: dict) -> None:
    _, eixos = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for eixo, (nome, curva) in zip(eixos, curvas.items()):
        _desenhar_curva(eixo, curva["tamanhos"], curva["treino"], curva["validacao"])
        eixo.set(title=nome, xlabel="nº de exemplos de treino", ylim=(.5, 1.0))
    eixos[0].set_ylabel("F1")
    eixos[0].legend()
    salvar_figura("05_curvas_aprendizado.png")


def matrizes_de_confusao_e_roc(modelos: dict, X_teste, y_teste, metricas: pd.DataFrame) -> None:
    _, eixos = plt.subplots(1, 4, figsize=(19, 4))
    for eixo, (nome, modelo) in zip(eixos[:3], modelos.items()):
        matriz = confusion_matrix(y_teste, modelo.predict(X_teste.values))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, ax=eixo,
                    xticklabels=["sem", "com"], yticklabels=["sem", "com"], annot_kws={"size": 14})
        eixo.set(title=nome, xlabel="previsto", ylabel="real")
    for nome, modelo in modelos.items():
        falso_positivo, verdadeiro_positivo, _ = roc_curve(y_teste, probabilidade_de_recorrencia(modelo, X_teste))
        eixos[3].plot(falso_positivo, verdadeiro_positivo, c=CORES_MODELOS[nome],
                      label=f"{nome} (AUC={metricas.loc[nome, 'roc_auc']:.2f})")
    eixos[3].plot([0, 1], [0, 1], "k--", lw=1)
    eixos[3].set(title="Curva ROC (teste)", xlabel="1 - especificidade", ylabel="recall")
    eixos[3].legend(fontsize=8, loc="lower right")
    salvar_figura("06_matrizes_confusao_roc.png")


def desenho_da_arvore(legivel) -> None:
    profundidade = legivel.arvore.get_depth()
    plt.figure(figsize=(max(13, 4 * profundidade + 4), 2.2 * profundidade + 2))
    plot_tree(legivel.arvore, feature_names=legivel.colunas_onehot, class_names=ROTULOS_CLASSES,
              filled=True, rounded=True, impurity=False, proportion=True, fontsize=10, precision=2)
    salvar_figura("07_arvore.png")


def importancia_da_arvore(importancia: pd.Series) -> None:
    plt.figure(figsize=(7, 5))
    importancia.plot.barh(color=CORES_MODELOS[ARVORE]).invert_yaxis()
    plt.xlabel("importância (redução de impureza)")
    plt.title("Árvore de Decisão - importância das variáveis")
    salvar_figura("08_arvore_importancia.png")


def probabilidades_categoricas_nb(tabela: pd.DataFrame, variaveis: list[str]) -> None:
    _, eixos = plt.subplots(2, 3, figsize=(16, 8))
    for eixo, coluna in zip(eixos.ravel(), variaveis):
        dados = tabela[tabela["variavel"] == coluna].set_index("categoria")[
            ["P(x|sem recorrência)", "P(x|com recorrência)"]]
        dados.plot.bar(ax=eixo, color=CORES_CLASSES, rot=20, legend=(coluna == variaveis[0]))
        eixo.set(title=coluna, xlabel="", ylabel="P(categoria | classe)", ylim=(0, 1))
    salvar_figura("09_nb_prob_categoricas.png")


def distribuicao_idade_nb(naive_bayes, X_treino: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 3.8))
    idades = np.linspace(X_treino["idade"].min() - 5, X_treino["idade"].max() + 5, 300)
    for classe, rotulo in enumerate(ROTULOS_CLASSES):
        media, desvio = naive_bayes.media_[classe, 0], naive_bayes.desvio_[classe, 0]
        densidade = norm.pdf(idades, media, desvio)
        plt.plot(idades, densidade, c=CORES_CLASSES[classe], label=f"{rotulo} (μ={media:.0f}, σ={desvio:.0f})")
        plt.fill_between(idades, densidade, alpha=.15, color=CORES_CLASSES[classe])
    plt.yticks([])
    plt.xlabel("idade")
    plt.legend(fontsize=9)
    plt.title("Naive Bayes - Normal da idade em cada classe")
    salvar_figura("10_nb_gaussiana_idade.png")


def importancia_permutacao(importancias: dict) -> None:
    _, eixos = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    for eixo, (nome, tabela) in zip(eixos, importancias.items()):
        tabela = tabela.sort_values("media")
        eixo.barh(tabela.index, tabela["media"], xerr=tabela["dp"], color=CORES_MODELOS[nome])
        eixo.axvline(0, c="k", lw=.8)
        eixo.set(title=nome, xlabel="queda na ROC-AUC ao embaralhar")
    salvar_figura("11_importancia_permutacao.png")


def shap_global(explicacoes: dict) -> None:
    for nome, explicacao in explicacoes.items():
        plt.figure()
        shap.plots.beeswarm(explicacao, max_display=12, show=False)
        plt.title(f"SHAP - {nome}")
        salvar_figura(f"12_shap_beeswarm_{SUFIXO_ARQUIVO_MODELO[nome]}.png")


def lime_do_knn(explicacao_lime) -> None:
    figura = explicacao_lime.as_pyplot_figure()
    figura.set_size_inches(8, 4.5)
    plt.title("LIME - KNN (caso exemplo): verde empurra para 'com recorrência'")
    salvar_figura("13_lime_knn_caso.png")


def shap_do_paciente(explicacoes: dict, posicao: int, descricao_paciente: dict) -> None:
    valores_legiveis = np.array([descricao_paciente[c] for c in ESQUEMA_PRINCIPAL.variaveis], dtype=object)
    for nome, explicacao in explicacoes.items():
        plt.figure()
        explicacao_paciente = shap.Explanation(values=explicacao.values[posicao],
                                               base_values=explicacao.base_values[posicao],
                                               data=valores_legiveis, feature_names=ESQUEMA_PRINCIPAL.variaveis)
        shap.plots.waterfall(explicacao_paciente, max_display=8, show=False)
        plt.title(f"SHAP - {nome} - caso exemplo")
        salvar_figura(f"14_shap_waterfall_{SUFIXO_ARQUIVO_MODELO[nome]}.png")


def contribuicoes_nb_do_paciente(contribuicoes: pd.Series, log_chance_priori: float, descricao_paciente: dict) -> None:
    plt.figure(figsize=(8, 5.5))
    ordenadas = contribuicoes.sort_values()
    plt.barh([f"{c} = {descricao_paciente[c]}" for c in ordenadas.index], ordenadas.values,
             color=[COR_DESTAQUE if valor > 0 else CORES_CLASSES[0] for valor in ordenadas.values])
    plt.axvline(0, c="k", lw=.8)
    plt.xlabel("log[P(x|com recorrência) / P(x|sem recorrência)]")
    plt.title(f"Naive Bayes - contribuição de cada variável (caso exemplo)\n"
              f"log-odds a priori = {log_chance_priori:+.2f}; total = {log_chance_priori + contribuicoes.sum():+.2f}")
    salvar_figura("15_nb_contribuicoes_caso.png")


def ranking_de_variaveis(posicoes: pd.DataFrame) -> None:
    ordem = posicoes.mean(axis=1).sort_values().index
    plt.figure(figsize=(10, 6.5))
    sns.heatmap(posicoes.loc[ordem], annot=True, cmap="viridis_r", cbar_kws={"label": "posição no ranking"})
    plt.title("Posição de cada variável no ranking de importância (1 = mais importante)")
    plt.xticks(rotation=30, ha="right")
    salvar_figura("16_ranking_comparado.png")


def correlacao_dos_rankings(correlacao: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(correlacao.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("Correlação de Spearman entre os rankings")
    plt.xticks(rotation=35, ha="right")
    salvar_figura("17_spearman_rankings.png")


def shap_comparado(importancia: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    importancia.loc[importancia.mean(axis=1).sort_values().index].plot.barh(
        color=[CORES_MODELOS[nome] for nome in importancia.columns], ax=plt.gca(), width=.8)
    plt.xlabel("média |SHAP| (impacto na probabilidade de recorrência)")
    plt.title("SHAP global - comparação entre os modelos")
    salvar_figura("18_shap_global_comparado.png")


def curva_de_calibracao(probabilidades: dict, y: np.ndarray, calibracao: pd.DataFrame) -> None:
    cores = dict(CORES_MODELOS, **{"Naive Bayes calibrado": COR_NB_CALIBRADO})
    plt.figure(figsize=(6.5, 5.5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="calibração perfeita")
    for nome, prob in probabilidades.items():
        proporcao_real, media_prevista = calibration_curve(y, prob, n_bins=6, strategy="quantile")
        plt.plot(media_prevista, proporcao_real, "o-", c=cores[nome],
                 label=f"{nome} (Brier {calibracao.loc[nome, 'brier']:.3f})")
    plt.xlabel("probabilidade prevista de recorrência")
    plt.ylabel("proporção real de recorrência")
    plt.title("Curva de calibração (validação cruzada)")
    plt.legend(fontsize=8, loc="upper left")
    salvar_figura("19_calibracao.png")


def recorrencia_por_sexo_e_fumo(tabela: pd.DataFrame) -> None:
    taxas = tabela.pivot(index="sexo", columns="fumante", values="taxa")[["não", "sim"]]
    quantidades = tabela.pivot(index="sexo", columns="fumante", values="n")[["não", "sim"]]
    plt.figure(figsize=(6.5, 4))
    eixo = taxas.plot.bar(rot=0, color=[COR_NEUTRA, COR_DESTAQUE], ax=plt.gca())
    for i, sexo in enumerate(taxas.index):
        for j, fumante in enumerate(taxas.columns):
            eixo.text(i + (j - .5) * .25, taxas.loc[sexo, fumante] + .02,
                      f"n={int(quantidades.loc[sexo, fumante])}", ha="center", fontsize=9)
    eixo.set(ylabel="taxa de recorrência", xlabel="", ylim=(0, 1), title="Recorrência por sexo e fumo")
    eixo.legend(title="fumante")
    salvar_figura("20_fumo_por_sexo.png")


def intervalo_das_razoes(intervalos: pd.DataFrame, pacientes_por_categoria: pd.Series) -> None:
    ordenado = intervalos.sort_values("razao")
    plt.figure(figsize=(8, 5))
    plt.errorbar(ordenado["razao"], range(len(ordenado)),
                 xerr=[ordenado["razao"] - ordenado["ic_2.5"], ordenado["ic_97.5"] - ordenado["razao"]],
                 fmt="o", c=CORES_MODELOS["KNN"], ecolor=COR_NEUTRA, capsize=3)
    plt.yticks(range(len(ordenado)), [f"{categoria}  (n={pacientes_por_categoria[categoria]})"
                                      for categoria in ordenado.index])
    plt.xscale("log")
    plt.axvline(1, c="k", lw=.8, ls="--")
    plt.xlabel("razão de verossimilhança (escala log) com IC 95% bootstrap")
    plt.title("Naive Bayes - quanto confiar em cada razão")
    salvar_figura("21_nb_ic_razoes.png")


def variacao_entre_divisoes(resultados: dict, metricas_teste: pd.DataFrame) -> None:
    coluna_no_teste = {"acuracia": "acuracia", "recall": "recall", "auc": "roc_auc"}
    nomes = list(resultados)
    _, eixos = plt.subplots(1, 3, figsize=(14, 4))
    for eixo, metrica in zip(eixos, ["acuracia", "recall", "auc"]):
        eixo.boxplot([resultados[nome][metrica] for nome in nomes], labels=["Árvore", "NB", "KNN"],
                     patch_artist=True, boxprops=dict(facecolor="#eef3f4"))
        for posicao, nome in enumerate(nomes):
            eixo.scatter(posicao + 1, metricas_teste.loc[nome, coluna_no_teste[metrica]], c=COR_DESTAQUE,
                         zorder=3, s=40, label="nosso teste" if posicao == 0 else None)
        eixo.set_title(metrica)
    eixos[0].legend(fontsize=8)
    plt.suptitle("Variação do desempenho em 50 divisões treino/teste diferentes")
    salvar_figura("22_variacao_50_divisoes.png")
