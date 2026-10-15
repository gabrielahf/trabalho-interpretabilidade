"""
Trabalho I - Interpretabilidade dos Modelos de Aprendizado de Máquina
=====================================================================

Problema: prever se um paciente com câncer diferenciado de tireoide terá
RECORRÊNCIA da doença, a partir de dados clínicos e patológicos do diagnóstico,
usando KNN, Naive Bayes e Árvore de Decisão. Depois investigamos POR QUE cada
modelo chega às suas previsões.

Dataset: UCI Differentiated Thyroid Cancer Recurrence (383 pacientes,
acompanhados por pelo menos 10 anos em um hospital no Irã).
Fonte: https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence

Como rodar:
    pip install -r requirements.txt
    python trabalho_interpretabilidade.py

O script baixa os dados (se não existirem em ./data), treina os modelos,
salva as figuras em ./figuras e as tabelas/métricas em ./resultados.
"""

import io
import json
import os
import urllib.request
import warnings
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import norm, spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DADOS = os.path.join(BASE, "data")
DIR_FIG = os.path.join(BASE, "figuras")
DIR_RES = os.path.join(BASE, "resultados")
for d in (DIR_DADOS, DIR_FIG, DIR_RES):
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
CORES = {"Árvore de Decisão": "#2a9d8f", "Naive Bayes": "#e76f51", "KNN": "#264653"}
COR_CLASSE = ["#2a9d8f", "#e76f51"]  # sem recorrência, com recorrência
CURTO = {"Árvore de Decisão": "arvore", "Naive Bayes": "nb", "KNN": "knn"}


def salvar_fig(nome):
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, nome), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# ETAPA 1 - Dados
# ---------------------------------------------------------------------------
URL = "https://archive.ics.uci.edu/static/public/915/differentiated+thyroid+cancer+recurrence.zip"
ARQ = os.path.join(DIR_DADOS, "Thyroid_Diff.csv")

# nomes das colunas em português
NOMES = {
    "Age": "idade", "Gender": "sexo", "Smoking": "fumante", "Hx Smoking": "hist_fumo",
    "Hx Radiothreapy": "hist_radioterapia", "Thyroid Function": "funcao_tireoide",
    "Physical Examination": "exame_fisico", "Adenopathy": "adenopatia",
    "Pathology": "patologia", "Focality": "focalidade", "Risk": "risco",
    "T": "T", "N": "N", "M": "M", "Stage": "estadio", "Response": "resposta_tratamento",
    "Recurred": "recorrencia",
}

# tradução das categorias e ORDEM delas (para variáveis ordinais como T, N,
# estádio e risco, a ordem segue a gravidade, o que deixa os gráficos legíveis)
CATEGORIAS = {
    "sexo": {"F": "feminino", "M": "masculino"},
    "fumante": {"No": "não", "Yes": "sim"},
    "hist_fumo": {"No": "não", "Yes": "sim"},
    "hist_radioterapia": {"No": "não", "Yes": "sim"},
    "funcao_tireoide": {"Euthyroid": "normal", "Clinical Hyperthyroidism": "hiper clínico",
                        "Subclinical Hyperthyroidism": "hiper subclínico",
                        "Clinical Hypothyroidism": "hipo clínico",
                        "Subclinical Hypothyroidism": "hipo subclínico"},
    "exame_fisico": {"Normal": "normal", "Single nodular goiter-left": "nódulo único esq.",
                     "Single nodular goiter-right": "nódulo único dir.",
                     "Multinodular goiter": "multinodular", "Diffuse goiter": "bócio difuso"},
    "adenopatia": {"No": "não", "Right": "direita", "Left": "esquerda", "Bilateral": "bilateral",
                   "Posterior": "posterior", "Extensive": "extensa"},
    "patologia": {"Micropapillary": "micropapilífero", "Papillary": "papilífero",
                  "Follicular": "folicular", "Hurthel cell": "células de Hürthle"},
    "focalidade": {"Uni-Focal": "unifocal", "Multi-Focal": "multifocal"},
    "risco": {"Low": "baixo", "Intermediate": "intermediário", "High": "alto"},
    "T": {v: v for v in ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"]},
    "N": {v: v for v in ["N0", "N1a", "N1b"]},
    "M": {v: v for v in ["M0", "M1"]},
    "estadio": {v: v for v in ["I", "II", "III", "IVA", "IVB"]},
    "resposta_tratamento": {"Excellent": "excelente", "Indeterminate": "indeterminada",
                            "Biochemical Incomplete": "bioquímica incompleta",
                            "Structural Incomplete": "estrutural incompleta"},
}


def carregar_dados():
    if not os.path.exists(ARQ):
        print("Baixando dataset da UCI...")
        conteudo = urllib.request.urlopen(URL).read()
        zipfile.ZipFile(io.BytesIO(conteudo)).extractall(DIR_DADOS)
    return pd.read_csv(ARQ).rename(columns=NOMES)


bruto = carregar_dados()
print(f"Instâncias: {len(bruto)} | colunas: {bruto.shape[1] - 1} features + alvo")
print("Valores ausentes:", int(bruto.isna().sum().sum()))
n_dup = int(bruto.duplicated().sum())
print("Linhas duplicadas:", n_dup)

# --- Decisões de limpeza --------------------------------------------------
#  * 19 linhas são idênticas a outras. Todas têm o perfil mais comum da base
#    (mulher jovem, risco baixo, T1/T2, N0, sem recorrência). Com só 16
#    atributos grosseiros, é esperado que pacientes diferentes tenham a mesma
#    ficha, então MANTEMOS essas linhas (removê-las distorceria a proporção das
#    classes). No fim do script há uma análise de sensibilidade sem elas.
#  * 'resposta_tratamento' é medida DEPOIS do tratamento, no acompanhamento.
#    "Resposta estrutural incompleta" significa, na prática, que a doença
#    voltou (89 de 91 recorreram). Usar essa variável seria vazamento temporal:
#    no momento do diagnóstico ela ainda não existe. Ela fica de fora dos
#    modelos e é usada só no experimento de vazamento no fim do script.
df = bruto.copy()
for c, mapa in CATEGORIAS.items():
    df[c] = df[c].map(mapa)
assert not df.isna().any().any(), "categoria sem tradução"
df["recorrencia"] = (df["recorrencia"] == "Yes").astype(int)
print(f"Recorrência: {df.recorrencia.mean():.1%}")

NUMERICAS = ["idade"]
CATEGORICAS = ["sexo", "fumante", "hist_fumo", "hist_radioterapia", "funcao_tireoide",
               "exame_fisico", "adenopatia", "patologia", "focalidade", "risco",
               "T", "N", "M", "estadio"]
FEATURES = NUMERICAS + CATEGORICAS
IDX_NUM = [0]
IDX_CAT = list(range(1, len(FEATURES)))
# rótulos de cada categoria na ordem do código inteiro (0, 1, 2, ...)
ROTULOS = {c: list(CATEGORIAS[c].values()) for c in CATEGORICAS}
N_CATS = [len(ROTULOS[c]) for c in CATEGORICAS]


def codificar(d, cols=CATEGORICAS):
    """Troca cada categoria pelo seu código inteiro (vocabulário fixo do dicionário de dados)."""
    X = d[NUMERICAS + cols].copy()
    for c in cols:
        X[c] = X[c].map({v: i for i, v in enumerate(list(CATEGORIAS[c].values()))}).astype(int)
    return X


# --- EDA ------------------------------------------------------------------
pd.Series({"instancias": len(bruto), "linhas_identicas": n_dup, "prop_recorrencia": round(df.recorrencia.mean(), 3)}
          ).to_csv(os.path.join(DIR_RES, "resumo_dados.csv"), header=["valor"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
df["recorrencia"].map({0: "sem recorrência", 1: "com recorrência"}).value_counts().plot.bar(
    ax=axes[0], color=COR_CLASSE, rot=0)
axes[0].set(title="Distribuição da classe", xlabel="")
sns.kdeplot(data=df, x="idade", hue="recorrencia", fill=True, common_norm=False,
            ax=axes[1], palette=COR_CLASSE)
axes[1].legend(["com recorrência", "sem recorrência"])
axes[1].set_title("Idade por classe")
salvar_fig("01_classe_e_idade.png")

# taxa de recorrência por categoria em algumas variáveis clínicas
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, c in zip(axes.ravel(), ["risco", "N", "adenopatia", "T", "focalidade", "sexo"]):
    taxa = df.groupby(c)["recorrencia"].agg(["mean", "size"]).reindex(ROTULOS[c]).dropna()
    ax.bar(taxa.index, taxa["mean"], color="#e76f51")
    for i, (m, n) in enumerate(zip(taxa["mean"], taxa["size"])):
        ax.text(i, m + .02, f"n={int(n)}", ha="center", fontsize=9)
    ax.set(title=c, ylim=(0, 1.1), ylabel="taxa de recorrência")
    ax.tick_params(axis="x", rotation=20)
salvar_fig("02_taxa_recorrencia_categorias.png")

# ---------------------------------------------------------------------------
# ETAPA 2 - Pré-processamento e treinamento
# ---------------------------------------------------------------------------
X_all = codificar(df)
y = df["recorrencia"].values

# divisão 80/20 estratificada (mantém a proporção de recorrência nos dois conjuntos)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.20, stratify=y, random_state=SEED)
X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
print(f"Treino: {len(X_tr)} | Teste: {len(X_te)}")


class NaiveBayesMisto(ClassifierMixin, BaseEstimator):
    """Naive Bayes que trata cada tipo de variável com a distribuição certa.

    P(y | x) ∝ P(y) * N(idade; μ_y, σ_y) * Π_cat P(x_j | y)

    Como é tudo um produto (soma em log), a contribuição de cada feature para
    a decisão pode ser lida diretamente, o que facilita a interpretação.
    """

    def __init__(self, alpha=1.0, idx_num=None, idx_cat=None, n_cats=None):
        self.alpha = alpha
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.n_cats = n_cats

    def _cods(self, X):
        X = np.asarray(X, dtype=float)
        c = np.rint(X[:, self.idx_cat]).astype(int)
        return np.clip(c, 0, np.array(self.n_cats) - 1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        Xn = X[:, self.idx_num]
        self.mu_ = np.array([Xn[y == c].mean(axis=0) for c in self.classes_])
        self.sigma_ = np.array([Xn[y == c].std(axis=0) for c in self.classes_]) + 1e-6
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.cnb_ = CategoricalNB(alpha=self.alpha, min_categories=self.n_cats).fit(self._cods(X), y)
        return self

    def contribuicoes_log(self, X):
        """log P(x_j | y) para cada amostra, feature e classe -> (n, n_feat, n_classes)."""
        X = np.asarray(X, dtype=float)
        out = np.zeros((X.shape[0], len(self.idx_num) + len(self.idx_cat), len(self.classes_)))
        for ci in range(len(self.classes_)):
            out[:, self.idx_num, ci] = norm.logpdf(X[:, self.idx_num], self.mu_[ci], self.sigma_[ci])
        cods = self._cods(X)
        for k, j in enumerate(self.idx_cat):
            out[:, j, :] = self.cnb_.feature_log_prob_[k][:, cods[:, k]].T
        return out

    def predict_proba(self, X):
        log_post = np.log(self.prior_) + self.contribuicoes_log(X).sum(axis=1)
        log_post -= log_post.max(axis=1, keepdims=True)
        p = np.exp(log_post)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def preproc_onehot(idx_num, idx_cat, n_cats):
    # KNN e Árvore: one-hot nas categóricas e padronização da idade. Para o KNN
    # é essencial (distância depende da escala); para a árvore não muda nada,
    # mas manter o mesmo pré-processamento simplifica a comparação.
    return ColumnTransformer([
        ("num", StandardScaler(), idx_num),
        ("cat", OneHotEncoder(categories=[list(range(n)) for n in n_cats], handle_unknown="ignore"), idx_cat),
    ])


def criar_modelos(idx_num, idx_cat, n_cats):
    return {
        "Árvore de Decisão": Pipeline([("prep", preproc_onehot(idx_num, idx_cat, n_cats)),
                                       ("clf", DecisionTreeClassifier(random_state=SEED))]),
        "Naive Bayes": Pipeline([("clf", NaiveBayesMisto(idx_num=idx_num, idx_cat=idx_cat, n_cats=n_cats))]),
        "KNN": Pipeline([("prep", preproc_onehot(idx_num, idx_cat, n_cats)), ("clf", KNeighborsClassifier())]),
    }


# grades de hiperparâmetros: todas limitam a complexidade para evitar overfitting
GRADES = {
    "Árvore de Decisão": {"clf__max_depth": [2, 3, 4, 5, 6, 8],
                          "clf__min_samples_leaf": [3, 5, 10, 20],
                          "clf__criterion": ["gini", "entropy"]},
    # alpha de 0,1 a 2 deu a mesma acurácia de CV (0,862 a 0,865); fixamos a
    # suavização de Laplace (alpha = 1), que evita probabilidades quase zero
    # em categorias raras e deixa as razões de verossimilhança mais legíveis.
    "Naive Bayes": {"clf__alpha": [1.0]},
    "KNN": {"clf__n_neighbors": list(range(3, 42, 2)),
            "clf__metric": ["euclidean", "manhattan"],
            "clf__weights": ["uniform"]},
}
# weights='distance' fica de fora de propósito: com ele o acerto no treino vira
# 100% (cada ponto é vizinho de si mesmo) e não dá para diagnosticar overfitting.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# critério: F1 da classe "com recorrência" (a classe minoritária e a mais importante)
modelos_base = criar_modelos(IDX_NUM, IDX_CAT, N_CATS)
modelos, melhores_params, buscas = {}, {}, {}
for nome, pipe in modelos_base.items():
    gs = GridSearchCV(pipe, GRADES[nome], cv=cv, scoring="f1", n_jobs=-1).fit(X_tr.values, y_tr)
    buscas[nome] = gs
    modelos[nome] = gs.best_estimator_
    melhores_params[nome] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    print(f"{nome}: {melhores_params[nome]} | F1 CV = {gs.best_score_:.3f}")

# Regra de parcimônia para a árvore: entre as configurações com F1 de CV a até
# 1 desvio-padrão da melhor, fica a mais rasa (mais simples de interpretar e
# menos propensa a overfitting).
res = pd.DataFrame(buscas["Árvore de Decisão"].cv_results_)
limite = res["mean_test_score"].max() - res.loc[res["rank_test_score"] == 1, "std_test_score"].iloc[0]
cands = res[res["mean_test_score"] >= limite].sort_values(
    ["param_clf__max_depth", "param_clf__min_samples_leaf", "mean_test_score"],
    ascending=[True, False, False])
escolhido = cands.iloc[0]["params"]
modelos["Árvore de Decisão"] = clone(modelos_base["Árvore de Decisão"]).set_params(**escolhido).fit(X_tr.values, y_tr)
melhores_params["Árvore de Decisão"] = {k.replace("clf__", ""): v for k, v in escolhido.items()}
print("Árvore (regra 1-DP):", melhores_params["Árvore de Decisão"])

with open(os.path.join(DIR_RES, "hiperparametros.json"), "w") as f:
    json.dump(melhores_params, f, indent=2, ensure_ascii=False, default=str)

# --- Curvas de validação: mostram onde começa o over/underfitting ---------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
prof = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
p_arv = {k: v for k, v in escolhido.items() if k not in ("clf__max_depth", "clf__min_samples_leaf")}
p_arv["clf__min_samples_leaf"] = 1  # sem limite de folha, para o efeito da profundidade aparecer
tr_s, va_s = validation_curve(clone(modelos_base["Árvore de Decisão"]).set_params(**p_arv), X_tr.values, y_tr,
                              param_name="clf__max_depth", param_range=prof, cv=cv, scoring="f1")
axes[0].plot(prof, tr_s.mean(1), "o-", label="treino")
axes[0].plot(prof, va_s.mean(1), "o-", label="validação (CV)")
axes[0].fill_between(prof, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=.2)
axes[0].axvline(escolhido["clf__max_depth"], ls="--", c="gray")
axes[0].set(xlabel="profundidade máxima", ylabel="F1", title="Árvore: profundidade")
axes[0].legend(loc="lower right")

ks = list(range(1, 80, 4))
p_knn = {"clf__metric": melhores_params["KNN"]["metric"]}
tr_s, va_s = validation_curve(clone(modelos_base["KNN"]).set_params(**p_knn), X_tr.values, y_tr,
                              param_name="clf__n_neighbors", param_range=ks, cv=cv, scoring="f1")
axes[1].plot(ks, tr_s.mean(1), "o-", label="treino")
axes[1].plot(ks, va_s.mean(1), "o-", label="validação (CV)")
axes[1].fill_between(ks, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=.2)
axes[1].axvline(melhores_params["KNN"]["n_neighbors"], ls="--", c="gray")
axes[1].set(xlabel="k (vizinhos)", ylabel="F1", title="KNN: número de vizinhos")
axes[1].legend(loc="upper right")
salvar_fig("04_curvas_validacao.png")

# valores usados no texto (árvore sem limite)
arv_livre = cross_validate(clone(modelos_base["Árvore de Decisão"]).set_params(clf__max_depth=None),
                           X_tr.values, y_tr, cv=cv, scoring="f1", return_train_score=True)

# --- Curvas de aprendizado -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (nome, m) in zip(axes, modelos.items()):
    tam, tr_s, va_s = learning_curve(m, X_tr.values, y_tr, cv=cv, scoring="f1",
                                     train_sizes=np.linspace(.2, 1, 8), random_state=SEED)
    ax.plot(tam, tr_s.mean(1), "o-", label="treino")
    ax.plot(tam, va_s.mean(1), "o-", label="validação (CV)")
    ax.fill_between(tam, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=.2)
    ax.set(title=nome, xlabel="nº de exemplos de treino", ylim=(.5, 1.0))
axes[0].set_ylabel("F1")
axes[0].legend()
salvar_fig("05_curvas_aprendizado.png")


# ---------------------------------------------------------------------------
# Avaliação: treino x validação cruzada x teste
# ---------------------------------------------------------------------------
def metricas(y_true, y_pred, y_prob):
    return {"acuracia": accuracy_score(y_true, y_pred),
            "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
            "precisao": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "especificidade": recall_score(y_true, y_pred, pos_label=0),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob)}


linhas, linhas_gap = [], []
for nome, m in modelos.items():
    p_te = m.predict_proba(X_te.values)[:, 1]
    met = metricas(y_te, (p_te >= .5).astype(int), p_te)
    linhas.append({"modelo": nome, **met})
    p_tr = m.predict_proba(X_tr.values)[:, 1]
    cvr = cross_validate(m, X_tr.values, y_tr, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    linhas_gap.append({"modelo": nome,
                       "acc_treino": accuracy_score(y_tr, (p_tr >= .5).astype(int)),
                       "acc_cv": cvr["test_accuracy"].mean(), "acc_cv_dp": cvr["test_accuracy"].std(),
                       "acc_teste": met["acuracia"],
                       "f1_treino": f1_score(y_tr, (p_tr >= .5).astype(int)),
                       "f1_cv": cvr["test_f1"].mean(), "f1_teste": met["f1"],
                       "auc_treino": roc_auc_score(y_tr, p_tr),
                       "auc_cv": cvr["test_roc_auc"].mean(), "auc_teste": met["roc_auc"]})

tab_met = pd.DataFrame(linhas).set_index("modelo").round(3)
tab_gap = pd.DataFrame(linhas_gap).set_index("modelo").round(3)
tab_met.to_csv(os.path.join(DIR_RES, "metricas_teste.csv"))
tab_gap.to_csv(os.path.join(DIR_RES, "treino_cv_teste.csv"))
print("\nMétricas no teste:\n", tab_met.to_string())
print("\nTreino x CV x Teste:\n", tab_gap.to_string())

# baseline: sempre chutar "sem recorrência"
baseline = max(y_te.mean(), 1 - y_te.mean())
print(f"Baseline (classe majoritária): acurácia {baseline:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(19, 4))
for ax, (nome, m) in zip(axes[:3], modelos.items()):
    cm = confusion_matrix(y_te, m.predict(X_te.values))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["sem", "com"], yticklabels=["sem", "com"], annot_kws={"size": 14})
    ax.set(title=nome, xlabel="previsto", ylabel="real")
for nome, m in modelos.items():
    fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te.values)[:, 1])
    axes[3].plot(fpr, tpr, label=f"{nome} (AUC={tab_met.loc[nome, 'roc_auc']:.2f})", c=CORES[nome])
axes[3].plot([0, 1], [0, 1], "k--", lw=1)
axes[3].set(title="Curva ROC (teste)", xlabel="1 - especificidade", ylabel="recall")
axes[3].legend(fontsize=8, loc="lower right")
salvar_fig("06_matrizes_confusao_roc.png")

# ---------------------------------------------------------------------------
# ETAPA 3a - Árvore de Decisão: estrutura + importância
# ---------------------------------------------------------------------------
arvore = modelos["Árvore de Decisão"].named_steps["clf"]
prep_arv = modelos["Árvore de Decisão"].named_steps["prep"]
nomes_oh = NUMERICAS + [f"{c}={r}" for c in CATEGORICAS for r in ROTULOS[c]]


def coluna_original(nome_oh):
    return nome_oh.split("=")[0]


# A árvore trabalha com a idade padronizada; para as regras ficarem legíveis,
# re-treinamos uma cópia idêntica SEM padronização (a árvore é invariante a
# escala, então os cortes são os mesmos, só que em anos).
def para_onehot_bruto(Xdf):
    return np.hstack([Xdf[NUMERICAS].values,
                      prep_arv.named_transformers_["cat"].transform(Xdf.values[:, IDX_CAT]).toarray()])


X_tr_oh = para_onehot_bruto(X_tr)
arvore_legivel = DecisionTreeClassifier(random_state=SEED, **melhores_params["Árvore de Decisão"]).fit(X_tr_oh, y_tr)
assert np.array_equal(arvore_legivel.predict(X_tr_oh), arvore.predict(prep_arv.transform(X_tr.values)))

regras = export_text(arvore_legivel, feature_names=nomes_oh, decimals=1, show_weights=True)
with open(os.path.join(DIR_RES, "arvore_regras.txt"), "w") as f:
    f.write(regras)
print("\nRegras da árvore:\n", regras)

prof_arv = arvore_legivel.get_depth()
plt.figure(figsize=(max(13, 4 * prof_arv + 4), 2.2 * prof_arv + 2))
plot_tree(arvore_legivel, feature_names=nomes_oh, class_names=["sem recorrência", "com recorrência"],
          filled=True, rounded=True, impurity=False, proportion=True, fontsize=10, precision=2)
salvar_fig("07_arvore.png")

imp_arv_oh = pd.Series(arvore_legivel.feature_importances_, index=nomes_oh)
imp_arv = imp_arv_oh.groupby(coluna_original).sum().reindex(FEATURES).sort_values(ascending=False)
imp_arv.to_csv(os.path.join(DIR_RES, "arvore_importancia_gini.csv"), header=["importancia"])
plt.figure(figsize=(7, 5))
imp_arv.plot.barh(color=CORES["Árvore de Decisão"]).invert_yaxis()
plt.xlabel("importância (redução de impureza)")
plt.title("Árvore de Decisão - importância das variáveis")
salvar_fig("08_arvore_importancia.png")

# ---------------------------------------------------------------------------
# ETAPA 3b - Naive Bayes: probabilidades condicionais
# ---------------------------------------------------------------------------
nb = modelos["Naive Bayes"].named_steps["clf"]
print(f"\nPrioris: P(sem)={nb.prior_[0]:.3f}  P(com)={nb.prior_[1]:.3f}")

linhas_cat = []
for k, c in enumerate(CATEGORICAS):
    probs = np.exp(nb.cnb_.feature_log_prob_[k])  # (2 classes, n categorias)
    for i, rot in enumerate(ROTULOS[c]):
        n_treino = int((X_tr[c] == i).sum())
        linhas_cat.append({"variavel": c, "categoria": rot, "n_treino": n_treino,
                           "P(x|sem recorrência)": probs[0, i], "P(x|com recorrência)": probs[1, i],
                           "razao_verossimilhanca": probs[1, i] / probs[0, i]})
tab_nb_cat = pd.DataFrame(linhas_cat).round(3)
tab_nb_cat.to_csv(os.path.join(DIR_RES, "nb_prob_condicionais_categoricas.csv"), index=False)
print(tab_nb_cat.to_string())

tab_nb_num = pd.DataFrame({"media_sem": nb.mu_[0], "dp_sem": nb.sigma_[0],
                           "media_com": nb.mu_[1], "dp_com": nb.sigma_[1]}, index=NUMERICAS).round(2)
tab_nb_num.to_csv(os.path.join(DIR_RES, "nb_gaussiana_idade.csv"))
print(tab_nb_num)

# Importância "global" do NB: quanto, em média, cada feature desloca o log da
# razão de chances (log P(x|com) - log P(x|sem)) nos pacientes de teste.
contrib_te = nb.contribuicoes_log(X_te.values)
llr_te = contrib_te[:, :, 1] - contrib_te[:, :, 0]
imp_nb = pd.Series(np.abs(llr_te).mean(axis=0), index=FEATURES).sort_values(ascending=False)
imp_nb.to_csv(os.path.join(DIR_RES, "nb_importancia_llr.csv"), header=["media_abs_log_razao"])

# gráfico das 6 categóricas que mais pesam no NB
top_nb_cat = [c for c in imp_nb.index if c in CATEGORICAS][:6]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, c in zip(axes.ravel(), top_nb_cat):
    t = tab_nb_cat[tab_nb_cat.variavel == c].set_index("categoria")[["P(x|sem recorrência)", "P(x|com recorrência)"]]
    t.plot.bar(ax=ax, color=COR_CLASSE, rot=20, legend=(c == top_nb_cat[0]))
    ax.set(title=c, xlabel="", ylabel="P(categoria | classe)", ylim=(0, 1))
salvar_fig("09_nb_prob_categoricas.png")

plt.figure(figsize=(7, 3.8))
xs = np.linspace(X_tr["idade"].min() - 5, X_tr["idade"].max() + 5, 300)
for ci, rot in enumerate(["sem recorrência", "com recorrência"]):
    dens = norm.pdf(xs, nb.mu_[ci, 0], nb.sigma_[ci, 0])
    plt.plot(xs, dens, c=COR_CLASSE[ci], label=f"{rot} (μ={nb.mu_[ci, 0]:.0f}, σ={nb.sigma_[ci, 0]:.0f})")
    plt.fill_between(xs, dens, alpha=.15, color=COR_CLASSE[ci])
plt.yticks([])
plt.xlabel("idade")
plt.legend(fontsize=9)
plt.title("Naive Bayes - Normal da idade em cada classe")
salvar_fig("10_nb_gaussiana_idade.png")

# ---------------------------------------------------------------------------
# ETAPA 3c - Ferramentas agnósticas: permutação e SHAP (para os 3 modelos)
# ---------------------------------------------------------------------------
perm = {}
for nome, m in modelos.items():
    r = permutation_importance(m, X_te.values, y_te, scoring="roc_auc", n_repeats=30, random_state=SEED)
    perm[nome] = pd.DataFrame({"media": r.importances_mean, "dp": r.importances_std}, index=FEATURES)
tab_perm = pd.concat({k: v["media"] for k, v in perm.items()}, axis=1).round(4)
tab_perm.to_csv(os.path.join(DIR_RES, "importancia_permutacao.csv"))

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
for ax, (nome, t) in zip(axes, perm.items()):
    t = t.sort_values("media")
    ax.barh(t.index, t["media"], xerr=t["dp"], color=CORES[nome])
    ax.axvline(0, c="k", lw=.8)
    ax.set(title=nome, xlabel="queda na ROC-AUC ao embaralhar")
salvar_fig("11_importancia_permutacao.png")

# SHAP (KernelExplainer): mesmo método para os 3 modelos, explicando a
# probabilidade de "com recorrência" no espaço das 15 variáveis originais.
fundo = shap.sample(X_tr, 60, random_state=SEED)
shap_vals = {}
for nome, m in modelos.items():
    f = (lambda mm: (lambda A: mm.predict_proba(np.asarray(A, dtype=float))[:, 1]))(m)
    expl = shap.KernelExplainer(f, fundo.values)
    sv = expl.shap_values(X_te.values, nsamples=500, silent=True)
    shap_vals[nome] = shap.Explanation(values=sv, base_values=np.full(len(X_te), expl.expected_value),
                                       data=X_te.values, feature_names=FEATURES)
    print(f"SHAP {nome} ok")

imp_shap = pd.DataFrame({n: np.abs(e.values).mean(0) for n, e in shap_vals.items()}, index=FEATURES)
imp_shap.round(4).to_csv(os.path.join(DIR_RES, "importancia_shap.csv"))

for nome, e in shap_vals.items():
    plt.figure()
    shap.plots.beeswarm(e, max_display=12, show=False)
    plt.title(f"SHAP - {nome}")
    salvar_fig(f"12_shap_beeswarm_{CURTO[nome]}.png")

# ---------------------------------------------------------------------------
# ETAPA 3d - Explicação local de um paciente (os três modelos)
# ---------------------------------------------------------------------------
# "Caso exemplo": paciente de teste COM recorrência em que os três modelos
# acertam, mas com probabilidades bem diferentes (bom para comparar).
prob_te = pd.DataFrame({n: m.predict_proba(X_te.values)[:, 1] for n, m in modelos.items()})
pred_te = (prob_te >= .5).astype(int)
acertam_todos = (pred_te.values == y_te[:, None]).all(axis=1) & (y_te == 1)
idx_caso = int(prob_te[acertam_todos].std(axis=1).idxmax())
caso = X_te.iloc[[idx_caso]]
caso_legivel = {c: (ROTULOS[c][int(caso[c].iloc[0])] if c in CATEGORICAS else int(caso[c].iloc[0])) for c in FEATURES}
print("\nCaso exemplo (índice no teste):", idx_caso, caso_legivel)
print(prob_te.iloc[idx_caso])

# concordância entre os modelos
concord = {f"{a} x {b}": float((pred_te[a] == pred_te[b]).mean())
           for i, a in enumerate(modelos) for b in list(modelos)[i + 1:]}
concord["os três"] = float((pred_te.nunique(axis=1) == 1).mean())
discord = pred_te.index[pred_te.nunique(axis=1) > 1]

# vizinhos mais próximos
knn_pipe = modelos["KNN"]
knn = knn_pipe.named_steps["clf"]
dist, viz = knn.kneighbors(knn_pipe.named_steps["prep"].transform(caso.values))
vizinhos = X_tr.iloc[viz[0]].copy()
for c in CATEGORICAS:
    vizinhos[c] = vizinhos[c].map(lambda v, c=c: ROTULOS[c][int(v)])
vizinhos = vizinhos.assign(recorrencia=y_tr[viz[0]], distancia=dist[0].round(2))
vizinhos.to_csv(os.path.join(DIR_RES, "knn_vizinhos_caso.csv"))
print(f"Vizinhos do caso: {vizinhos.recorrencia.sum()} de {len(vizinhos)} com recorrência")

# LIME: as categóricas já são códigos 0..n-1, então entram direto
NOMES_CAT_LIME = {j: ROTULOS[c] for j, c in zip(IDX_CAT, CATEGORICAS)}


def criar_lime(semente):
    return LimeTabularExplainer(X_tr.values.astype(float), feature_names=FEATURES,
                                class_names=["sem recorrência", "com recorrência"],
                                categorical_features=IDX_CAT, categorical_names=NOMES_CAT_LIME,
                                discretize_continuous=True, random_state=semente)


lime_exp = criar_lime(SEED)
lime_res = {}
for nome, m in modelos.items():
    e = lime_exp.explain_instance(caso.values[0].astype(float), m.predict_proba, num_features=8, num_samples=5000)
    lime_res[nome] = e.as_list()
    if nome == "KNN":
        fig = e.as_pyplot_figure()
        fig.set_size_inches(8, 4.5)
        plt.title("LIME - KNN (caso exemplo): verde empurra para 'com recorrência'")
        salvar_fig("13_lime_knn_caso.png")
with open(os.path.join(DIR_RES, "lime_caso.json"), "w") as f:
    json.dump(lime_res, f, indent=2, ensure_ascii=False)

# estabilidade do LIME: repete com sementes diferentes e vê se o top-3 muda
tops = []
for s in range(10):
    e = criar_lime(s).explain_instance(caso.values[0].astype(float), knn_pipe.predict_proba,
                                       num_features=3, num_samples=1000)
    tops.append(tuple(sorted(n for n, _ in e.as_map()[1])))
estab_lime = pd.Series(tops).value_counts().iloc[0] / len(tops)

for nome, e in shap_vals.items():
    plt.figure()
    ex = shap.Explanation(values=e.values[idx_caso], base_values=e.base_values[idx_caso],
                          data=np.array([caso_legivel[c] for c in FEATURES], dtype=object),
                          feature_names=FEATURES)
    shap.plots.waterfall(ex, max_display=8, show=False)
    plt.title(f"SHAP - {nome} - caso exemplo")
    salvar_fig(f"14_shap_waterfall_{CURTO[nome]}.png")
shap_caso = pd.DataFrame({n: e.values[idx_caso] for n, e in shap_vals.items()}, index=FEATURES).round(3)
shap_caso.to_csv(os.path.join(DIR_RES, "shap_caso.csv"))

# caminho do caso na árvore
caso_oh = para_onehot_bruto(caso)
t_ = arvore_legivel.tree_
caminho = []
for n in arvore_legivel.decision_path(caso_oh).indices:
    if t_.children_left[n] == -1:
        v = t_.value[n][0]
        v = v / v.sum()
        caminho.append(f"FOLHA: {v.round(3).tolist()} (proporção sem/com)")
    else:
        f_, thr = nomes_oh[t_.feature[n]], t_.threshold[n]
        v = caso_oh[0, t_.feature[n]]
        caminho.append(f"{f_} = {v:.1f} {'<=' if v <= thr else '>'} {thr:.1f}")
with open(os.path.join(DIR_RES, "arvore_caminho_caso.txt"), "w") as f:
    f.write("\n".join(caminho))
print("Caminho na árvore:", *caminho, sep="\n  ")

# decomposição exata do NB para o caso (log da razão de chances por feature)
contrib_caso = nb.contribuicoes_log(caso.values)[0]
llr_caso = pd.Series(contrib_caso[:, 1] - contrib_caso[:, 0], index=FEATURES)
log_prior = np.log(nb.prior_[1] / nb.prior_[0])
plt.figure(figsize=(8, 5.5))
o = llr_caso.sort_values()
plt.barh([f"{c} = {caso_legivel[c]}" for c in o.index], o.values,
         color=["#e76f51" if v > 0 else "#2a9d8f" for v in o.values])
plt.axvline(0, c="k", lw=.8)
plt.xlabel("log[P(x|com recorrência) / P(x|sem recorrência)]")
plt.title(f"Naive Bayes - contribuição de cada variável (caso exemplo)\n"
          f"log-odds a priori = {log_prior:+.2f}; total = {log_prior + llr_caso.sum():+.2f}")
salvar_fig("15_nb_contribuicoes_caso.png")
llr_caso.round(3).to_csv(os.path.join(DIR_RES, "nb_contribuicoes_caso.csv"), header=["log_razao"])

# ---------------------------------------------------------------------------
# ETAPA 4 - Comparação: os modelos concordam sobre as variáveis importantes?
# ---------------------------------------------------------------------------
ranking = pd.DataFrame({
    "Árvore (Gini)": imp_arv,
    "NB (|log-razão| média)": imp_nb,
    "Árvore (permutação)": tab_perm["Árvore de Decisão"],
    "NB (permutação)": tab_perm["Naive Bayes"],
    "KNN (permutação)": tab_perm["KNN"],
    "Árvore (SHAP)": imp_shap["Árvore de Decisão"],
    "NB (SHAP)": imp_shap["Naive Bayes"],
    "KNN (SHAP)": imp_shap["KNN"],
}).reindex(FEATURES)
postos = ranking.rank(ascending=False, method="min").astype(int)
postos.to_csv(os.path.join(DIR_RES, "ranking_variaveis.csv"))
ordem = postos.mean(axis=1).sort_values().index
plt.figure(figsize=(10, 6.5))
sns.heatmap(postos.loc[ordem], annot=True, cmap="viridis_r", cbar_kws={"label": "posição no ranking"})
plt.title("Posição de cada variável no ranking de importância (1 = mais importante)")
plt.xticks(rotation=30, ha="right")
salvar_fig("16_ranking_comparado.png")

corr_sp = pd.DataFrame(index=ranking.columns, columns=ranking.columns, dtype=float)
for a in ranking.columns:
    for b in ranking.columns:
        corr_sp.loc[a, b] = spearmanr(ranking[a], ranking[b])[0]
corr_sp.round(2).to_csv(os.path.join(DIR_RES, "spearman_rankings.csv"))
plt.figure(figsize=(8, 6.5))
sns.heatmap(corr_sp.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
plt.title("Correlação de Spearman entre os rankings")
plt.xticks(rotation=35, ha="right")
salvar_fig("17_spearman_rankings.png")

plt.figure(figsize=(9, 6))
(imp_shap.loc[imp_shap.mean(axis=1).sort_values().index]
 .plot.barh(color=[CORES[c] for c in imp_shap.columns], ax=plt.gca(), width=.8))
plt.xlabel("média |SHAP| (impacto na probabilidade de recorrência)")
plt.title("SHAP global - comparação entre os modelos")
salvar_fig("18_shap_global_comparado.png")

# ---------------------------------------------------------------------------
# Experimento de vazamento: e se usássemos 'resposta_tratamento'?
# ---------------------------------------------------------------------------
# Mostra que a métrica sobe, mas o modelo passa a depender de uma informação
# que só existe depois do tratamento, ou seja, ele "olha o futuro".
CAT_V = CATEGORICAS + ["resposta_tratamento"]
Xv = codificar(df, CAT_V)
Xv_tr, Xv_te = train_test_split(Xv, test_size=0.20, stratify=y, random_state=SEED)
mods_v = criar_modelos(IDX_NUM, list(range(1, len(CAT_V) + 1)), [len(CATEGORIAS[c]) for c in CAT_V])
linhas_v = []
for nome, pipe in mods_v.items():
    gs = GridSearchCV(pipe, GRADES[nome], cv=cv, scoring="f1", n_jobs=-1).fit(Xv_tr.values, y_tr)
    p = gs.best_estimator_.predict_proba(Xv_te.values)[:, 1]
    r = permutation_importance(gs.best_estimator_, Xv_te.values, y_te, scoring="roc_auc",
                               n_repeats=20, random_state=SEED)
    imp_v = pd.Series(r.importances_mean, index=NUMERICAS + CAT_V).sort_values(ascending=False)
    linhas_v.append({"modelo": nome, "acuracia": accuracy_score(y_te, p >= .5), "f1": f1_score(y_te, p >= .5),
                     "roc_auc": roc_auc_score(y_te, p), "variavel_mais_importante": imp_v.index[0],
                     "queda_auc_top1": imp_v.iloc[0]})
tab_vaz = pd.DataFrame(linhas_v).set_index("modelo").round(3)
tab_vaz.to_csv(os.path.join(DIR_RES, "experimento_vazamento_resposta.csv"))
print("\nCom 'resposta_tratamento' (vazamento):\n", tab_vaz.to_string())

# ---------------------------------------------------------------------------
# Sensibilidade: e se as 19 linhas idênticas fossem removidas?
# ---------------------------------------------------------------------------
dedup = df.drop_duplicates()
X_dd, y_dd = codificar(dedup).values, dedup["recorrencia"].values
cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
sens = {}
for nome, m in modelos.items():
    sens[nome] = {"acc_cv_com_identicas": float(cross_validate(clone(m), X_all.values, y, cv=cv_rep, scoring="accuracy")["test_score"].mean()),
                  "acc_cv_sem_identicas": float(cross_validate(clone(m), X_dd, y_dd, cv=cv_rep, scoring="accuracy")["test_score"].mean())}
tab_sens = pd.DataFrame(sens).T.round(3)
tab_sens.to_csv(os.path.join(DIR_RES, "sensibilidade_linhas_identicas.csv"))
print("\nSensibilidade (validação cruzada 5x5 na base inteira):\n", tab_sens.to_string())

# ---------------------------------------------------------------------------
# ETAPA 5 - Verificações de robustez (os 4 pontos de cuidado da análise)
# ---------------------------------------------------------------------------
# Probabilidades "fora da amostra" para a base inteira: cada paciente recebe a
# previsão de um modelo que NÃO o viu no treino (validação cruzada 5 partições).
# Os hiperparâmetros são os escolhidos acima.
cv_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
nb_calibrado = CalibratedClassifierCV(clone(modelos["Naive Bayes"]), method="sigmoid", cv=5)
modelos_cal = dict(modelos)
modelos_cal["Naive Bayes calibrado"] = nb_calibrado
oof = {nome: cross_val_predict(clone(m), X_all.values, y, cv=cv_oof, method="predict_proba")[:, 1]
       for nome, m in modelos_cal.items()}

# --- 5.1 Calibração: o NB é confiante demais? ------------------------------
linhas_cal = []
for nome, p in oof.items():
    extremos = (p > .99) | (p < .01)
    acerto_extremos = ((p[extremos] >= .5).astype(int) == y[extremos]).mean() if extremos.any() else np.nan
    linhas_cal.append({"modelo": nome, "brier": brier_score_loss(y, p),
                       "acuracia_oof": accuracy_score(y, p >= .5), "auc_oof": roc_auc_score(y, p),
                       "pct_prob_extrema": extremos.mean(), "acerto_nas_extremas": acerto_extremos})
tab_cal = pd.DataFrame(linhas_cal).set_index("modelo").round(3)
tab_cal.to_csv(os.path.join(DIR_RES, "robustez_1_calibracao.csv"))
print("\n[5.1] Calibração (probabilidades fora da amostra):\n", tab_cal.to_string())

nb_cal_split = clone(nb_calibrado).fit(X_tr.values, y_tr)
p_caso_cal = float(nb_cal_split.predict_proba(caso.values)[0, 1])
print(f"Caso exemplo: NB {prob_te.loc[idx_caso, 'Naive Bayes']:.5f} -> NB calibrado {p_caso_cal:.3f}")

plt.figure(figsize=(6.5, 5.5))
plt.plot([0, 1], [0, 1], "k--", lw=1, label="calibração perfeita")
cores_cal = dict(CORES, **{"Naive Bayes calibrado": "#e9c46a"})
for nome, p in oof.items():
    frac, media = calibration_curve(y, p, n_bins=6, strategy="quantile")
    plt.plot(media, frac, "o-", c=cores_cal[nome], label=f"{nome} (Brier {tab_cal.loc[nome, 'brier']:.3f})")
plt.xlabel("probabilidade prevista de recorrência")
plt.ylabel("proporção real de recorrência")
plt.title("Curva de calibração (validação cruzada)")
plt.legend(fontsize=8, loc="upper left")
salvar_fig("19_calibracao.png")

# --- 5.2 Fumar ou ser homem? -------------------------------------------------
estrat = (df.groupby(["sexo", "fumante"])["recorrencia"].agg(taxa="mean", n="size").reset_index())
estrat["taxa"] = estrat["taxa"].round(3)
estrat.to_csv(os.path.join(DIR_RES, "robustez_2_fumo_por_sexo.csv"), index=False)
print("\n[5.2] Recorrência por sexo e fumo:\n", estrat.to_string(index=False))
pct_fumantes_homens = float((df.loc[df.fumante == "sim", "sexo"] == "masculino").mean())

cat_sem_fumo = [c for c in CATEGORICAS if c != "fumante"]
X_sem_fumo = codificar(df, cat_sem_fumo)
mods_sf = criar_modelos(IDX_NUM, list(range(1, len(cat_sem_fumo) + 1)), [len(ROTULOS[c]) for c in cat_sem_fumo])
linhas_fumo = []
for nome, m in modelos.items():
    p_sf = mods_sf[nome].set_params(**{f"clf__{k}": v for k, v in melhores_params[nome].items()})
    com = cross_validate(clone(m), X_all.values, y, cv=cv_rep, scoring=["accuracy", "roc_auc"])
    sem = cross_validate(p_sf, X_sem_fumo.values, y, cv=cv_rep, scoring=["accuracy", "roc_auc"])
    linhas_fumo.append({"modelo": nome, "acc_com_fumo": com["test_accuracy"].mean(),
                        "acc_sem_fumo": sem["test_accuracy"].mean(),
                        "auc_com_fumo": com["test_roc_auc"].mean(), "auc_sem_fumo": sem["test_roc_auc"].mean()})
tab_fumo = pd.DataFrame(linhas_fumo).set_index("modelo").round(3)
tab_fumo.to_csv(os.path.join(DIR_RES, "robustez_2_modelos_sem_fumo.csv"))
print("Modelos sem a variável fumante (CV 5x5):\n", tab_fumo.to_string())

plt.figure(figsize=(6.5, 4))
piv = estrat.pivot(index="sexo", columns="fumante", values="taxa")[["não", "sim"]]
nn = estrat.pivot(index="sexo", columns="fumante", values="n")[["não", "sim"]]
ax = piv.plot.bar(rot=0, color=["#8d99ae", "#e76f51"], ax=plt.gca())
for i, s in enumerate(piv.index):
    for j, f in enumerate(piv.columns):
        ax.text(i + (j - .5) * .25, piv.loc[s, f] + .02, f"n={int(nn.loc[s, f])}", ha="center", fontsize=9)
ax.set(ylabel="taxa de recorrência", xlabel="", ylim=(0, 1), title="Recorrência por sexo e fumo")
ax.legend(title="fumante")
salvar_fig("20_fumo_por_sexo.png")

# --- 5.3 Categorias raras: o quanto as razões dependem da suavização? -------
destaque = [("risco", "alto"), ("M", "M1"), ("T", "T4a"), ("estadio", "IVB"), ("adenopatia", "extensa"),
            ("adenopatia", "bilateral"), ("N", "N1b"), ("fumante", "sim"), ("sexo", "masculino"),
            ("risco", "baixo"), ("patologia", "micropapilífero")]


def razoes(nb_model, pares):
    out = {}
    for c, rot in pares:
        k, i = CATEGORICAS.index(c), ROTULOS[c].index(rot)
        lp = nb_model.cnb_.feature_log_prob_[k][:, i]
        out[f"{c}={rot}"] = float(np.exp(lp[1] - lp[0]))
    return out


nb_base = modelos["Naive Bayes"].named_steps["clf"]
tab_alpha = pd.DataFrame({f"alpha={a}": razoes(clone(nb_base).set_params(alpha=a).fit(X_tr.values, y_tr), destaque)
                          for a in [0.1, 0.5, 1.0, 2.0]})
tab_alpha.insert(0, "n_treino", [int((X_tr[c] == ROTULOS[c].index(r)).sum()) for c, r in destaque])
tab_alpha.insert(1, "n_com_recorrencia", [int(((X_tr[c] == ROTULOS[c].index(r)) & (y_tr == 1)).sum()) for c, r in destaque])
tab_alpha.round(2).to_csv(os.path.join(DIR_RES, "robustez_3_razoes_vs_alpha.csv"))
print("\n[5.3] Razão de verossimilhança conforme a suavização:\n", tab_alpha.round(2).to_string())

acc_alpha = {a: float(cross_validate(clone(modelos["Naive Bayes"]).set_params(clf__alpha=a), X_all.values, y,
                                     cv=cv_rep, scoring="accuracy")["test_score"].mean()) for a in [0.1, 0.5, 1.0, 2.0]}
print("Acurácia CV 5x5 por alpha:", {k: round(v, 3) for k, v in acc_alpha.items()})

# intervalo de confiança (bootstrap) das razões com alpha = 1
rng = np.random.RandomState(SEED)
boot = []
for _ in range(500):
    ix = rng.randint(0, len(X_tr), len(X_tr))
    boot.append(razoes(clone(nb_base).fit(X_tr.values[ix], y_tr[ix]), destaque))
boot = pd.DataFrame(boot)
ic = pd.DataFrame({"razao": tab_alpha["alpha=1.0"], "ic_2.5": boot.quantile(.025), "ic_97.5": boot.quantile(.975)})
ic.round(2).to_csv(os.path.join(DIR_RES, "robustez_3_ic_bootstrap.csv"))
print("IC 95% bootstrap (alpha = 1):\n", ic.round(2).to_string())

plt.figure(figsize=(8, 5))
o = ic.sort_values("razao")
plt.errorbar(o["razao"], range(len(o)), xerr=[o["razao"] - o["ic_2.5"], o["ic_97.5"] - o["razao"]],
             fmt="o", c="#264653", ecolor="#8d99ae", capsize=3)
plt.yticks(range(len(o)), [f"{i}  (n={n})" for i, n in zip(o.index, tab_alpha.loc[o.index, "n_treino"])])
plt.xscale("log")
plt.axvline(1, c="k", lw=.8, ls="--")
plt.xlabel("razão de verossimilhança (escala log) com IC 95% bootstrap")
plt.title("Naive Bayes - quanto confiar em cada razão")
salvar_fig("21_nb_ic_razoes.png")

# --- 5.4 Generalização: variação entre divisões, subgrupos e estabilidade ----
cv50 = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)  # 50 testes de ~77 pacientes
dist = {}
for nome, m in modelos.items():
    r = cross_validate(clone(m), X_all.values, y, cv=cv50, scoring=["accuracy", "recall", "roc_auc"])
    dist[nome] = pd.DataFrame({"acuracia": r["test_accuracy"], "recall": r["test_recall"], "auc": r["test_roc_auc"]})
tab_dist = pd.concat({n: d.describe(percentiles=[.025, .975]).T[["mean", "2.5%", "97.5%"]] for n, d in dist.items()})
tab_dist.round(3).to_csv(os.path.join(DIR_RES, "robustez_4_50_divisoes.csv"))
print("\n[5.4] 50 divisões treino/teste (média e faixa de 95%):\n", tab_dist.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, met in zip(axes, ["acuracia", "recall", "auc"]):
    ax.boxplot([dist[n][met] for n in modelos], labels=["Árvore", "NB", "KNN"], patch_artist=True,
               boxprops=dict(facecolor="#eef3f4"))
    for i, n in enumerate(modelos):
        ax.scatter(i + 1, tab_met.loc[n, {"acuracia": "acuracia", "recall": "recall", "auc": "roc_auc"}[met]],
                   c="#e76f51", zorder=3, s=40, label="nosso teste" if i == 0 else None)
    ax.set_title(met)
axes[0].legend(fontsize=8)
plt.suptitle("Variação do desempenho em 50 divisões treino/teste diferentes")
salvar_fig("22_variacao_50_divisoes.png")

grupos = {"mulheres": df.sexo.values == "feminino", "homens": df.sexo.values == "masculino",
          "idade < 55": df.idade.values < 55, "idade >= 55": df.idade.values >= 55}
linhas_sub = []
for g, mask in grupos.items():
    for nome in modelos:
        p = oof[nome][mask]
        yy = y[mask]
        linhas_sub.append({"grupo": g, "n": int(mask.sum()), "prop_recorrencia": yy.mean(), "modelo": nome,
                           "acuracia": accuracy_score(yy, p >= .5), "recall": recall_score(yy, p >= .5)})
tab_sub = pd.DataFrame(linhas_sub).round(3)
tab_sub.to_csv(os.path.join(DIR_RES, "robustez_4_subgrupos.csv"), index=False)
print("Desempenho por subgrupo (fora da amostra):\n",
      tab_sub.pivot(index=["grupo", "n", "prop_recorrencia"], columns="modelo", values=["acuracia", "recall"]).to_string())

raizes, conjuntos = [], []
for _ in range(200):
    ix = rng.randint(0, len(X_tr), len(X_tr))
    t = DecisionTreeClassifier(random_state=SEED, **melhores_params["Árvore de Decisão"]).fit(X_tr_oh[ix], y_tr[ix])
    usadas = sorted({coluna_original(nomes_oh[f]) for f in t.tree_.feature if f >= 0})
    raizes.append(coluna_original(nomes_oh[t.tree_.feature[0]]))
    conjuntos.append(" + ".join(usadas))
estab_arv = {"raiz": pd.Series(raizes).value_counts(normalize=True).round(3).to_dict(),
             "variaveis_usadas": pd.Series(conjuntos).value_counts(normalize=True).round(3).head(5).to_dict()}
print("Estabilidade da árvore (200 bootstraps):", estab_arv)

# --- 5.5 Divisão 70/30 x 80/20 ----------------------------------------------
# Repetimos TODO o processo (divisão, busca de hiperparâmetros e teste) com 30
# sementes para cada proporção. Comparar uma única divisão de cada seria
# decidir pela sorte de quem cai no teste. Leva cerca de 2 minutos.
linhas_div = []
for ts in [0.2, 0.3]:
    for semente in range(30):
        Xa, Xb, ya, yb = train_test_split(X_all.values, y, test_size=ts, stratify=y, random_state=semente)
        cvs = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
        for nome, pipe in criar_modelos(IDX_NUM, IDX_CAT, N_CATS).items():
            gs = GridSearchCV(pipe, GRADES[nome], cv=cvs, scoring="f1", n_jobs=-1).fit(Xa, ya)
            p = gs.best_estimator_.predict_proba(Xb)[:, 1]
            linhas_div.append({"divisao": f"{round((1 - ts) * 100)}/{round(ts * 100)}", "semente": semente,
                               "modelo": nome, "acc_teste": accuracy_score(yb, p >= .5),
                               "recall": recall_score(yb, p >= .5), "f1": f1_score(yb, p >= .5),
                               "auc": roc_auc_score(yb, p), "acc_treino": gs.best_estimator_.score(Xa, ya)})
tab_div = pd.DataFrame(linhas_div)
tab_div.to_csv(os.path.join(DIR_RES, "robustez_5_divisao_80_20_vs_70_30.csv"), index=False)
resumo_div = tab_div.groupby(["modelo", "divisao"]).agg(
    acc=("acc_teste", "mean"), acc_dp=("acc_teste", "std"), recall=("recall", "mean"),
    f1=("f1", "mean"), auc=("auc", "mean")).round(3)
print("\n[5.5] 80/20 x 70/30 (30 sementes cada):\n", resumo_div.to_string())

robustez = {
    "calibracao": tab_cal.to_dict(orient="index"),
    "caso_nb_calibrado": p_caso_cal,
    "fumo_por_sexo": estrat.to_dict(orient="records"),
    "pct_fumantes_homens": pct_fumantes_homens,
    "modelos_sem_fumo": tab_fumo.to_dict(orient="index"),
    "razoes_vs_alpha": tab_alpha.round(2).to_dict(orient="index"),
    "acc_cv_por_alpha": acc_alpha,
    "ic_razoes": ic.round(2).to_dict(orient="index"),
    "variacao_50_divisoes": {f"{a}|{b}": v for (a, b), v in tab_dist.round(3)["mean"].items()},
    "variacao_50_divisoes_faixa": {f"{a}|{b}": [r["2.5%"], r["97.5%"]] for (a, b), r in tab_dist.round(3).iterrows()},
    "subgrupos": linhas_sub,
    "estabilidade_arvore": estab_arv,
    "divisao_80_20_vs_70_30": {f"{a}|{b}": r for (a, b), r in resumo_div.to_dict(orient="index").items()},
}
with open(os.path.join(DIR_RES, "robustez.json"), "w") as f:
    json.dump(robustez, f, indent=2, ensure_ascii=False, default=float)

resumo = {
    "linhas_identicas": n_dup, "sensibilidade_identicas": sens,
    "n_instancias": int(len(df)), "n_treino": int(len(X_tr)), "n_teste": int(len(X_te)),
    "prop_recorrencia": float(y.mean()), "baseline_acuracia": float(baseline),
    "hiperparametros": melhores_params,
    "metricas_teste": tab_met.to_dict(orient="index"),
    "treino_cv_teste": tab_gap.to_dict(orient="index"),
    "arvore_sem_limite_f1": {"treino": float(arv_livre["train_score"].mean()),
                             "cv": float(arv_livre["test_score"].mean())},
    "concordancia_previsoes": concord, "n_discordancias_teste": int(len(discord)),
    "caso_exemplo": {"indice_teste": idx_caso, "valores": caso_legivel,
                     "probabilidades": prob_te.iloc[idx_caso].to_dict()},
    "knn_vizinhos_recorrencia": f"{int(vizinhos.recorrencia.sum())}/{len(vizinhos)}",
    "lime_estabilidade_top3": float(estab_lime),
    "nb_prioris": nb.prior_.tolist(),
    "folhas_arvore": int(arvore_legivel.get_n_leaves()),
    "profundidade_arvore": int(arvore_legivel.get_depth()),
    "experimento_vazamento": tab_vaz.to_dict(orient="index"),
}
with open(os.path.join(DIR_RES, "resumo.json"), "w") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False, default=float)
print("\nConcordância:", concord)
print("Estabilidade LIME (top-3 igual):", estab_lime)
print("\nFim. Figuras em ./figuras, tabelas em ./resultados")
