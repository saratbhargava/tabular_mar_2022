import os

from hyperopt import hp
from hyperopt.pyll.base import scope


DEVICE = "cpu" # "cpu", "cuda"

TRAINING_FILE = "../input/train.csv"

TESTING_FILE = "../input/test.csv"

MODELS = "../models/"

SUBMIT = "../submit/"

N_SPLITS = 5

RANDOM_STATE= 42

VALIDATION_TYPE = "StratifiedKfold" # "StratifiedKfold", "Kfold"

TARGET_LABEL = "target"


# Hyper parameter search with hyperopt
hyper_params = {
    "decision_tree": {
        "criterion": hp.choice("criteria", ["gini", "entropy"]),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
    },
    "rf": {
        "n_estimators": hp.choice("n_estimators", [50, 100, 200, 500, 700, 1000, 1200, 1300, 1500]),
        "criterion": hp.choice("criteria", ["gini", "entropy"]),
        "n_jobs": -1,
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    },
    "svm": {
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        # "loss": hp.choice("loss", ["hinge"]),
        "C": hp.loguniform("C", -2, 8),
    },
    "xgb":{
        "learning_rate": hp.choice("learning_rate", [0.5]),
        "n_estimators": hp.choice("n_estimators", [150, 175]),
        "max_depth": hp.choice("max_depth", [10, 15, 20]),
        "reg_lambda": hp.choice("reg_lambda", [7.5, 10, 12.5]),
        "reg_alpha": hp.choice("reg_alpha", [0.1, 0.5, 1.0]),
        "min_split_loss": hp.choice("min_split_loss", [0.1, 0.2]),
        "n_jobs": -1,
        "tree_method": "gpu_hist",
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
    }
}

fixed_hyper_params = {
    "decision_tree": {
        "criterion": "entropy", # "entropy", "gini"
        "min_samples_split": 2, # 2-10
    },
    "rf": {
        "n_estimators": 100,
        "criterion": "entropy", # "entropy", "gini"
        "n_jobs": -1,
        "min_samples_split": 3, # 2-10
    },
    "svm": {
        "penalty": "l2",
        "C": 2,
    },
    "xgb":{
        "learning_rate": 0.5,
        "n_estimators": 100,
        "max_depth": 15,
        "reg_lambda": 10.0,
        "reg_alpha": 0.5,
        "min_split_loss": 0.1,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "tree_method": "gpu_hist",
    },
    "autosklearn":{
    }
}

# wandb settings
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_SILENT'] = 'true'
PROJECT="Tabular_Feb2022"
ENTITY="sarat"

# dashboard section
INDEX_NAME = "sample_idx"
TARGET_NAME = "Bacteria"
DASHBOARD_PORT = 8052
