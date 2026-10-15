import warnings

import numpy as np
import pandas as pd

from src import avaliacao, graficos, interpretabilidade, robustez
from src.config import ARVORE, COLUNA_ALVO, KNN, NAIVE_BAYES, SEMENTE
from src.dados import (ESQUEMA_PRINCIPAL, codificar_variaveis, carregar_dataset_original,
                       descrever_paciente, dividir_treino_teste, resumir_dataset, traduzir_dataset)
from src.modelos import treinar_modelos
from src.saida import (criar_pastas_de_saida, imprimir_titulo, salvar_json, salvar_tabela,
                       salvar_texto)

warnings.filterwarnings("ignore")


def etapa_1_dados():
    """Carrega, traduz e descreve o dataset."""
    imprimir_titulo("Etapa 1: dados")
    dataset_original = carregar_dataset_original()
    dataset = traduzir_dataset(dataset_original)
    resumo = resumir_dataset(dataset_original, dataset)
    print(resumo.to_string())
    salvar_tabela(resumo, "resumo_dados.csv", header=["valor"])
    graficos.classe_e_idade(dataset)
    graficos.taxa_de_recorrencia_por_categoria(dataset)
    return dataset_original, dataset


def etapa_2_treinamento(dataset):
    """Divide os dados, treina os três modelos e avalia o desempenho."""
    imprimir_titulo("Etapa 2: pré-processamento, treinamento e avaliação")
    X = codificar_variaveis(dataset)
    y = dataset[COLUNA_ALVO].values
    divisao = dividir_treino_teste(X, y)
    print(f"Treino: {len(divisao.X_treino)} | Teste: {len(divisao.X_teste)}")

    modelos, hiperparametros, modelos_base = treinar_modelos(
        ESQUEMA_PRINCIPAL, divisao.X_treino.values, divisao.y_treino)
    salvar_json(hiperparametros, "hiperparametros.json")

    graficos.curvas_de_validacao(avaliacao.curva_profundidade_arvore(modelos_base, hiperparametros, divisao),
                                 avaliacao.curva_vizinhos_knn(modelos_base, hiperparametros, divisao))
    arvore_sem_limite = avaliacao.avaliar_arvore_sem_limite(modelos_base, divisao)
    graficos.curvas_de_aprendizado(avaliacao.curvas_de_aprendizado(modelos, divisao))

    metricas = avaliacao.avaliar_no_teste(modelos, divisao)
    treino_validacao_teste = avaliacao.comparar_treino_validacao_teste(modelos, divisao)
    salvar_tabela(metricas, "metricas_teste.csv")
    salvar_tabela(treino_validacao_teste, "treino_cv_teste.csv")
    referencia = avaliacao.acuracia_de_referencia(divisao.y_teste)
    print("\nMétricas no teste:\n", metricas.to_string())
    print("\nTreino x validação x teste:\n", treino_validacao_teste.to_string())
    print(f"Acurácia chutando a classe majoritária: {referencia:.3f}")
    graficos.matrizes_de_confusao_e_roc(modelos, divisao.X_teste, divisao.y_teste, metricas)

    return {"X": X, "y": y, "divisao": divisao, "modelos": modelos, "hiperparametros": hiperparametros,
            "metricas": metricas, "treino_validacao_teste": treino_validacao_teste,
            "arvore_sem_limite": arvore_sem_limite, "referencia": referencia}


def etapa_3_interpretabilidade(treino):
    """Explica cada modelo e o caso exemplo."""
    divisao, modelos = treino["divisao"], treino["modelos"]

    imprimir_titulo("Etapa 3a: árvore de decisão")
    arvore_legivel = interpretabilidade.criar_arvore_legivel(
        modelos[ARVORE], treino["hiperparametros"][ARVORE], divisao)
    regras = interpretabilidade.regras_da_arvore(arvore_legivel)
    importancia_arvore = interpretabilidade.importancia_gini(arvore_legivel)
    print(regras)
    salvar_texto(regras, "arvore_regras.txt")
    salvar_tabela(importancia_arvore, "arvore_importancia_gini.csv", header=["importancia"])
    graficos.desenho_da_arvore(arvore_legivel)
    graficos.importancia_da_arvore(importancia_arvore)

    imprimir_titulo("Etapa 3b: naive bayes")
    naive_bayes = modelos[NAIVE_BAYES].named_steps["modelo"]
    print(f"Prioris: P(sem) = {naive_bayes.priori_[0]:.3f} | P(com) = {naive_bayes.priori_[1]:.3f}")
    probabilidades_nb = interpretabilidade.probabilidades_condicionais(naive_bayes, divisao.X_treino)
    distribuicao_idade = interpretabilidade.distribuicao_da_idade(naive_bayes)
    importancia_nb = interpretabilidade.importancia_naive_bayes(naive_bayes, divisao.X_teste)
    print(probabilidades_nb.to_string())
    salvar_tabela(probabilidades_nb, "nb_prob_condicionais_categoricas.csv", index=False)
    salvar_tabela(distribuicao_idade, "nb_gaussiana_idade.csv")
    salvar_tabela(importancia_nb, "nb_importancia_llr.csv", header=["media_abs_log_razao"])
    variaveis_mais_importantes = [c for c in importancia_nb.index if c in ESQUEMA_PRINCIPAL.categoricas][:6]
    graficos.probabilidades_categoricas_nb(probabilidades_nb, variaveis_mais_importantes)
    graficos.distribuicao_idade_nb(naive_bayes, divisao.X_treino)

    imprimir_titulo("Etapa 3c: permutação e SHAP nos três modelos")
    permutacao = interpretabilidade.importancia_por_permutacao(modelos, divisao)
    salvar_tabela(interpretabilidade.tabela_permutacao(permutacao), "importancia_permutacao.csv")
    graficos.importancia_permutacao(permutacao)
    explicacoes_shap = interpretabilidade.calcular_shap(modelos, divisao)
    shap_medio = interpretabilidade.importancia_shap(explicacoes_shap)
    salvar_tabela(shap_medio.round(4), "importancia_shap.csv")
    graficos.shap_global(explicacoes_shap)

    imprimir_titulo("Etapa 3d: caso exemplo")
    probabilidades = interpretabilidade.probabilidades_no_teste(modelos, divisao)
    posicao = interpretabilidade.escolher_caso_exemplo(probabilidades, divisao.y_teste)
    paciente = divisao.X_teste.iloc[[posicao]]
    descricao = descrever_paciente(divisao.X_teste, posicao)
    print(f"Paciente {posicao} do teste: {descricao}")
    print(probabilidades.iloc[posicao].to_string())
    concordancia, discordancias = interpretabilidade.concordancia_entre_modelos(probabilidades)

    vizinhos = interpretabilidade.vizinhos_do_paciente(modelos[KNN], paciente, divisao)
    salvar_tabela(vizinhos, "knn_vizinhos_caso.csv")
    print(f"Vizinhos com recorrência: {vizinhos['recorrencia'].sum()} de {len(vizinhos)}")

    explicacoes_lime = interpretabilidade.explicar_com_lime(modelos, paciente, divisao.X_treino)
    salvar_json({nome: explicacao.as_list() for nome, explicacao in explicacoes_lime.items()}, "lime_caso.json")
    graficos.lime_do_knn(explicacoes_lime[KNN])
    estabilidade_lime = interpretabilidade.estabilidade_do_lime(modelos[KNN], paciente, divisao.X_treino)

    graficos.shap_do_paciente(explicacoes_shap, posicao, descricao)
    salvar_tabela(interpretabilidade.shap_do_paciente(explicacoes_shap, posicao), "shap_caso.csv")

    caminho = interpretabilidade.caminho_na_arvore(arvore_legivel, paciente)
    salvar_texto("\n".join(caminho), "arvore_caminho_caso.txt")
    print("Caminho na árvore:", *caminho, sep="\n  ")

    contribuicoes, log_chance_priori = interpretabilidade.contribuicoes_naive_bayes_paciente(naive_bayes, paciente)
    graficos.contribuicoes_nb_do_paciente(contribuicoes, log_chance_priori, descricao)
    salvar_tabela(contribuicoes.round(3), "nb_contribuicoes_caso.csv", header=["log_razao"])

    return {"arvore_legivel": arvore_legivel, "naive_bayes": naive_bayes, "importancia_arvore": importancia_arvore,
            "importancia_nb": importancia_nb, "permutacao": permutacao, "shap_medio": shap_medio,
            "probabilidades": probabilidades, "posicao": posicao, "paciente": paciente, "descricao": descricao,
            "concordancia": concordancia, "discordancias": discordancias, "vizinhos": vizinhos,
            "estabilidade_lime": estabilidade_lime}


def etapa_4_comparacao(explicacao):
    """Compara as importâncias das variáveis entre os modelos e as ferramentas."""
    imprimir_titulo("Etapa 4: comparação")
    rankings = interpretabilidade.montar_rankings(explicacao["importancia_arvore"], explicacao["importancia_nb"],
                                                  explicacao["permutacao"], explicacao["shap_medio"])
    posicoes = rankings.rank(ascending=False, method="min").astype(int)
    correlacao = interpretabilidade.correlacao_entre_rankings(rankings)
    salvar_tabela(posicoes, "ranking_variaveis.csv")
    salvar_tabela(correlacao.round(2), "spearman_rankings.csv")
    graficos.ranking_de_variaveis(posicoes)
    graficos.correlacao_dos_rankings(correlacao)
    graficos.shap_comparado(explicacao["shap_medio"])
    print("Concordância entre modelos:", explicacao["concordancia"])


def etapa_5_robustez(dataset, treino, explicacao):
    """Testa os pontos frágeis da análise."""
    X, y, divisao, modelos = treino["X"], treino["y"], treino["divisao"], treino["modelos"]

    imprimir_titulo("Experimento de vazamento (com resposta ao tratamento)")
    vazamento = robustez.experimento_vazamento(dataset, y)
    salvar_tabela(vazamento, "experimento_vazamento_resposta.csv")
    print(vazamento.to_string())

    imprimir_titulo("Sensibilidade às linhas idênticas")
    sensibilidade = robustez.sensibilidade_linhas_identicas(modelos, dataset, X, y)
    salvar_tabela(sensibilidade.round(3), "sensibilidade_linhas_identicas.csv")
    print(sensibilidade.round(3).to_string())

    imprimir_titulo("Etapa 5.1: calibração")
    fora_da_amostra = robustez.probabilidades_fora_da_amostra(modelos, X, y)
    calibracao = robustez.analisar_calibracao(fora_da_amostra, y)
    salvar_tabela(calibracao, "robustez_1_calibracao.csv")
    nb_calibrado_paciente = robustez.probabilidade_calibrada_do_paciente(
        modelos[NAIVE_BAYES], divisao, explicacao["paciente"])
    print(calibracao.to_string())
    print(f"Caso exemplo: NB {explicacao['probabilidades'].loc[explicacao['posicao'], NAIVE_BAYES]:.5f}"
          f" -> NB calibrado {nb_calibrado_paciente:.3f}")
    graficos.curva_de_calibracao(fora_da_amostra, y, calibracao)

    imprimir_titulo("Etapa 5.2: fumo e sexo")
    fumo_sexo = robustez.recorrencia_por_sexo_e_fumo(dataset)
    sem_fumo = robustez.modelos_sem_fumo(modelos, treino["hiperparametros"], dataset, X, y)
    salvar_tabela(fumo_sexo, "robustez_2_fumo_por_sexo.csv", index=False)
    salvar_tabela(sem_fumo, "robustez_2_modelos_sem_fumo.csv")
    print(fumo_sexo.to_string(index=False))
    print(sem_fumo.to_string())
    graficos.recorrencia_por_sexo_e_fumo(fumo_sexo)

    imprimir_titulo("Etapa 5.3: categorias raras")
    gerador = np.random.RandomState(SEMENTE)
    razoes = robustez.razoes_por_suavizacao(explicacao["naive_bayes"], divisao)
    salvar_tabela(razoes.round(2), "robustez_3_razoes_vs_alpha.csv")
    acuracia_alpha = robustez.acuracia_por_suavizacao(modelos[NAIVE_BAYES], X, y)
    intervalos = robustez.intervalo_confianca_razoes(explicacao["naive_bayes"], divisao, gerador,
                                                     razoes["alpha=1.0"])
    salvar_tabela(intervalos.round(2), "robustez_3_ic_bootstrap.csv")
    print(razoes.round(2).to_string())
    print("Acurácia por alpha:", {alpha: round(valor, 3) for alpha, valor in acuracia_alpha.items()})
    graficos.intervalo_das_razoes(intervalos, razoes["n_treino"])

    imprimir_titulo("Etapa 5.4: generalização")
    variacao = robustez.variacao_entre_divisoes(modelos, X, y)
    resumo_variacao = robustez.resumir_variacao(variacao)
    salvar_tabela(resumo_variacao.round(3), "robustez_4_50_divisoes.csv")
    print(resumo_variacao.round(3).to_string())
    graficos.variacao_entre_divisoes(variacao, treino["metricas"])
    subgrupos = robustez.desempenho_por_subgrupo(fora_da_amostra, dataset, y, list(modelos))
    salvar_tabela(pd.DataFrame(subgrupos).round(3), "robustez_4_subgrupos.csv", index=False)
    estabilidade_arvore = robustez.estabilidade_da_arvore(explicacao["arvore_legivel"], treino["hiperparametros"],
                                                          divisao, gerador)
    print("Estabilidade da árvore:", estabilidade_arvore)

    imprimir_titulo("Etapa 5.5: divisão 80/20 x 70/30")
    divisoes = robustez.comparar_proporcoes_de_divisao(X, y)
    resumo_divisoes = robustez.resumir_proporcoes_de_divisao(divisoes)
    salvar_tabela(divisoes, "robustez_5_divisao_80_20_vs_70_30.csv", index=False)
    print(resumo_divisoes.to_string())

    salvar_json({
        "calibracao": calibracao.to_dict(orient="index"),
        "caso_nb_calibrado": nb_calibrado_paciente,
        "fumo_por_sexo": fumo_sexo.to_dict(orient="records"),
        "pct_fumantes_homens": robustez.proporcao_de_homens_entre_fumantes(dataset),
        "modelos_sem_fumo": sem_fumo.to_dict(orient="index"),
        "razoes_vs_alpha": razoes.round(2).to_dict(orient="index"),
        "acc_cv_por_alpha": acuracia_alpha,
        "ic_razoes": intervalos.round(2).to_dict(orient="index"),
        "variacao_50_divisoes": {f"{a}|{b}": v for (a, b), v in resumo_variacao.round(3)["mean"].items()},
        "variacao_50_divisoes_faixa": {f"{a}|{b}": [linha["2.5%"], linha["97.5%"]]
                                       for (a, b), linha in resumo_variacao.round(3).iterrows()},
        "subgrupos": subgrupos,
        "estabilidade_arvore": estabilidade_arvore,
        "divisao_80_20_vs_70_30": {f"{a}|{b}": linha for (a, b), linha in resumo_divisoes.to_dict(orient="index").items()},
    }, "robustez.json")
    return {"vazamento": vazamento, "sensibilidade": sensibilidade}


def salvar_resumo(dataset_original, dataset, treino, explicacao, verificacoes):
    divisao = treino["divisao"]
    salvar_json({
        "linhas_identicas": int(dataset_original.duplicated().sum()),
        "sensibilidade_identicas": verificacoes["sensibilidade"].to_dict(orient="index"),
        "n_instancias": len(dataset), "n_treino": len(divisao.X_treino), "n_teste": len(divisao.X_teste),
        "prop_recorrencia": float(treino["y"].mean()),
        "baseline_acuracia": treino["referencia"],
        "hiperparametros": treino["hiperparametros"],
        "metricas_teste": treino["metricas"].to_dict(orient="index"),
        "treino_cv_teste": treino["treino_validacao_teste"].to_dict(orient="index"),
        "arvore_sem_limite_f1": treino["arvore_sem_limite"],
        "concordancia_previsoes": explicacao["concordancia"],
        "n_discordancias_teste": explicacao["discordancias"],
        "caso_exemplo": {"indice_teste": explicacao["posicao"], "valores": explicacao["descricao"],
                         "probabilidades": explicacao["probabilidades"].iloc[explicacao["posicao"]].to_dict()},
        "knn_vizinhos_recorrencia": f"{int(explicacao['vizinhos']['recorrencia'].sum())}/{len(explicacao['vizinhos'])}",
        "lime_estabilidade_top3": explicacao["estabilidade_lime"],
        "nb_prioris": explicacao["naive_bayes"].priori_.tolist(),
        "folhas_arvore": int(explicacao["arvore_legivel"].arvore.get_n_leaves()),
        "profundidade_arvore": int(explicacao["arvore_legivel"].arvore.get_depth()),
        "experimento_vazamento": verificacoes["vazamento"].to_dict(orient="index"),
    }, "resumo.json")


def main():
    np.random.seed(SEMENTE)
    criar_pastas_de_saida()
    dataset_original, dataset = etapa_1_dados()
    treino = etapa_2_treinamento(dataset)
    explicacao = etapa_3_interpretabilidade(treino)
    etapa_4_comparacao(explicacao)
    verificacoes = etapa_5_robustez(dataset, treino, explicacao)
    salvar_resumo(dataset_original, dataset, treino, explicacao, verificacoes)
    print("\nFim. Figuras em figuras/ e tabelas em resultados/.")


if __name__ == "__main__":
    main()
