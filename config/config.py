from datetime import datetime

# ===================================================
# DATA PATHs
# ===================================================
DATABASE = "mimic_iii"
MODEL = "ctabgan"
TODAY = datetime.today().strftime("%Y-%m-%d")
BUCKET_NAME = "syn-aws-par-s3-common-dev-bucket"

PATH_RAW_DATA = f"raw_data/{DATABASE}/"
PATH_TEMP_DATA = f"temp_data/{MODEL}/"

# output data
PATH_OUTPUT_DATA = f"output_data/{MODEL}/"
PATH_SYNTH_DATA = f"output_data/{MODEL}/"
PATH_MODEL = PATH_OUTPUT_DATA + "models/"
PATH_EVALUATE = f"output_data/{MODEL}/evaluate/"

FILE_PREPARED_DATA = f"prepared_data/{TODAY}_{MODEL}_{DATABASE}_prepared_data.csv"
FILE_PREPROC_DATA = (
    f"preprocessed_data/{TODAY}_{MODEL}_{DATABASE}_preprocessed_data.csv"
)
FILE_SYNTHESIZED_DATA = (
    f"synthesized_data/{TODAY}_{MODEL}_{DATABASE}_synthesized_data.csv"
)
FILE_MODEL = f"{TODAY}_{MODEL}_{DATABASE}.pkl"

# =================================================
# preprocessing
# =================================================
PATH_METADATA = f"output_data/{MODEL}/metadata/"
FILE_METADATA = f"metadata_{MODEL}_{DATABASE}.txt"

# =========
# Modelling
# =========
RANDOM_STATE = 123

# bnt
FILE_BN_CONSTRAINTS = "bnt/bnt_constraints.csv"
SCORING_FUNC = "BIC"  # BIC, AIC, K2

# copulas
COPULA_TYPE = "gaussian"  # gaussian or vine
VINE_TYPE = "regular"  # only if vine, can be ‘center’,’direct’,’regular’
FILE_COPULA_MODEL = "{}_copulas.sav".format(COPULA_TYPE)

# CTABGAN
CONFIG_CTABGAN = {
    "CATEGORICAL_COLUMNS": ["GENDER", "ETHNICITY"],
    "LOG_COLUMNS": ["FOLLOWUP_PERIOD"],
    "MIXED_COLUMNS": {"AGE_AT_LAST_ADMISSION": [0.0]},
    "INTEGER_COLUMNS": ["AGE_AT_LAST_ADMISSION", "FOLLOWUP_PERIOD"],
    "PROBLEM_TYPE": {
        "Classification": "GENDER"
    },  # other possibility for CTABGAN+ for MIMIC_III : {"Regression": "AGE_AT_LAST_ADMISSION"}
}
CTABGAN_TEST_RATIO = 0.20

# =========
# Sampling
# =========
N_SAMPLE = 1000  # NB: could be add to parser

# =========
# Evaluating
# =========

# ==== fidelity evaluating ====
METRICS_TO_COMPUTE = [
        "WD",
        "JSD",
        "KSComplement",
        "TVComplement",
        "CorrelationSimilarity",
        "ContingencySimilarity",
    ]

# ==== utility evaluating =====
# Define performance metrics to evaluate models
PERF_METRICS = ["accuracy", "recall", "precision", "roc_auc", "average_precision"]

# Reference metric for models ranking
BEST_PERF_METRIC = "average_precision"

# Models to fit and their hyperparameters
MODELS_PARAMS = {
    "lgbm": {
        "default_hyperparameters": {
            "boosting_type": "gbdt",
            "objective": "binary",
            "random_state": 10,
            "feature_fraction_seed": 10,
            "force_col_wise": True,
            "deterministic": True,
        },
        "grid_hyperparameters": {
            "n_estimators": [100, 300],
            "num_leaves": [8, 31],
            "colsample_bytree": [0.5, 1.0],
            "max_depth": [2, 4, 8],
            "learning_rate": [0.1, 0.01],
            "min_child_samples": [10, 20],
            "reg_alpha": [0.0, 10],
            "reg_lambda": [0.0, 10],
            "is_unbalance": [True, False],
        },
    },
    "rf": {
        "default_hyperparameters": {"random_state": 10, "n_jobs": 1},
        "grid_hyperparameters": {
            "n_estimators": [100, 300],
            "max_depth": [6, 8],
            "min_samples_leaf": [0.005, 0.01],
            "class_weight": [None, "balanced"],
        },
    },
    "lr": {
        "grid_hyperparameters": {"C": [0.5, 1, 1.5], "penalty": ["l2"]},
        "default_hyperparameters": {"random_state": 10},
    },
}

# Grid search n_cv
N_CV = 3

# Possibility to detail the score by fold
SPLIT_PERF = False

# perform shap analysis to better understand what feature is not well synthesised
SHAP = False
