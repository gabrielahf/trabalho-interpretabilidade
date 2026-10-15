"""Configurações do projeto: caminhos, colunas do dataset, hiperparâmetros e cores."""

from pathlib import Path

SEMENTE = 42

PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_RAIZ / "data"
PASTA_FIGURAS = PASTA_RAIZ / "figuras"
PASTA_RESULTADOS = PASTA_RAIZ / "resultados"

URL_DATASET = "https://archive.ics.uci.edu/static/public/915/differentiated+thyroid+cancer+recurrence.zip"
ARQUIVO_DATASET = PASTA_DADOS / "Thyroid_Diff.csv"

# Nome original da coluna -> nome usado no projeto
NOMES_COLUNAS = {
    "Age": "idade",
    "Gender": "sexo",
    "Smoking": "fumante",
    "Hx Smoking": "historico_fumo",
    "Hx Radiothreapy": "historico_radioterapia",
    "Thyroid Function": "funcao_tireoide",
    "Physical Examination": "exame_fisico",
    "Adenopathy": "adenopatia",
    "Pathology": "patologia",
    "Focality": "focalidade",
    "Risk": "risco",
    "T": "T",
    "N": "N",
    "M": "M",
    "Stage": "estadio",
    "Response": "resposta_tratamento",
    "Recurred": "recorrencia",
}

# Tradução das categorias. A ordem define o código numérico de cada uma (0, 1, 2...).
# Em T, N, estádio e risco, a ordem segue a gravidade.
CATEGORIAS = {
    "sexo": {"F": "feminino", "M": "masculino"},
    "fumante": {"No": "não", "Yes": "sim"},
    "historico_fumo": {"No": "não", "Yes": "sim"},
    "historico_radioterapia": {"No": "não", "Yes": "sim"},
    "funcao_tireoide": {
        "Euthyroid": "normal",
        "Clinical Hyperthyroidism": "hiper clínico",
        "Subclinical Hyperthyroidism": "hiper subclínico",
        "Clinical Hypothyroidism": "hipo clínico",
        "Subclinical Hypothyroidism": "hipo subclínico",
    },
    "exame_fisico": {
        "Normal": "normal",
        "Single nodular goiter-left": "nódulo único esq.",
        "Single nodular goiter-right": "nódulo único dir.",
        "Multinodular goiter": "multinodular",
        "Diffuse goiter": "bócio difuso",
    },
    "adenopatia": {
        "No": "não", "Right": "direita", "Left": "esquerda",
        "Bilateral": "bilateral", "Posterior": "posterior", "Extensive": "extensa",
    },
    "patologia": {
        "Micropapillary": "micropapilífero", "Papillary": "papilífero",
        "Follicular": "folicular", "Hurthel cell": "células de Hürthle",
    },
    "focalidade": {"Uni-Focal": "unifocal", "Multi-Focal": "multifocal"},
    "risco": {"Low": "baixo", "Intermediate": "intermediário", "High": "alto"},
    "T": {valor: valor for valor in ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"]},
    "N": {valor: valor for valor in ["N0", "N1a", "N1b"]},
    "M": {valor: valor for valor in ["M0", "M1"]},
    "estadio": {valor: valor for valor in ["I", "II", "III", "IVA", "IVB"]},
    "resposta_tratamento": {
        "Excellent": "excelente",
        "Indeterminate": "indeterminada",
        "Biochemical Incomplete": "bioquímica incompleta",
        "Structural Incomplete": "estrutural incompleta",
    },
}

COLUNA_ALVO = "recorrencia"

# Medida depois do tratamento: fica fora dos modelos e só entra no experimento de vazamento
COLUNA_VAZAMENTO = "resposta_tratamento"

VARIAVEIS_NUMERICAS = ["idade"]
VARIAVEIS_CATEGORICAS = [
    "sexo", "fumante", "historico_fumo", "historico_radioterapia", "funcao_tireoide",
    "exame_fisico", "adenopatia", "patologia", "focalidade", "risco",
    "T", "N", "M", "estadio",
]

ROTULOS_CLASSES = ["sem recorrência", "com recorrência"]

PROPORCAO_TESTE = 0.20
NUMERO_PARTICOES_CV = 5
LIMIAR_DECISAO = 0.5

ARVORE = "Árvore de Decisão"
NAIVE_BAYES = "Naive Bayes"
KNN = "KNN"

# Grades de busca. O prefixo "modelo__" se refere à etapa "modelo" do Pipeline.
GRADES_HIPERPARAMETROS = {
    ARVORE: {
        "modelo__max_depth": [2, 3, 4, 5, 6, 8],
        "modelo__min_samples_leaf": [3, 5, 10, 20],
        "modelo__criterion": ["gini", "entropy"],
    },
    # alpha entre 0,1 e 2 deu a mesma acurácia; alpha = 1 evita probabilidades quase zero
    NAIVE_BAYES: {"modelo__alpha": [1.0]},
    # sem weights="distance", que faria o acerto no treino ser sempre 100%
    KNN: {
        "modelo__n_neighbors": list(range(3, 42, 2)),
        "modelo__metric": ["euclidean", "manhattan"],
        "modelo__weights": ["uniform"],
    },
}

CORES_MODELOS = {ARVORE: "#2a9d8f", NAIVE_BAYES: "#e76f51", KNN: "#264653"}
COR_NB_CALIBRADO = "#e9c46a"
CORES_CLASSES = ["#2a9d8f", "#e76f51"]  # sem recorrência, com recorrência
COR_DESTAQUE = "#e76f51"
COR_NEUTRA = "#8d99ae"
SUFIXO_ARQUIVO_MODELO = {ARVORE: "arvore", NAIVE_BAYES: "nb", KNN: "knn"}
