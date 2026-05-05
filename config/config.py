from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent

config = {
    "path": {
        "raw": {
            "train": str(PROJECT_ROOT / "data" / "raw" / "train.csv"),
            "test":  str(PROJECT_ROOT / "data" / "raw" / "test.csv"),
        },
        "processed": {
            "train": str(PROJECT_ROOT / "data" / "processed" / "train.pkl"),
            "test":  str(PROJECT_ROOT / "data" / "processed" / "test.pkl"),
        },
        "submissions": str(PROJECT_ROOT / "data" / "submissions"),
        "models":      str(PROJECT_ROOT / "models"),
    },
    "data": {
        "target":           "SalePrice",
        "id_col":           "Id",
        "target_transform": "log1p",
    },
    "features": {
        "skew_threshold": 0.75,
    },
    "cv": {
        "n_folds":      5,
        "random_state": 42,
    },
    "models": {
        "ridge": {
            "alphas": [0.1, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 75.0, 100.0],
        },
        "lasso": {
            "alphas":    [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            "max_iter":  5000,
        },
        "elasticnet": {
            "alphas":    [0.001, 0.01, 0.1, 0.5],
            "l1_ratios": [0.1, 0.3, 0.5, 0.7, 0.9],
            "max_iter":  5000,
        },
        "lgbm": {
            "n_estimators":          5000,
            "learning_rate":         0.01,
            "num_leaves":            31,
            "min_child_samples":     20,
            "subsample":             0.8,
            "colsample_bytree":      0.8,
            "reg_alpha":             0.1,
            "reg_lambda":            0.1,
            "random_state":          42,
            "early_stopping_rounds": 200,
        },
        "xgb": {
            "n_estimators":          5000,
            "learning_rate":         0.01,
            "max_depth":             4,
            "min_child_weight":      2,
            "subsample":             0.8,
            "colsample_bytree":      0.8,
            "gamma":                 0.1,
            "reg_alpha":             0.1,
            "reg_lambda":            0.5,
            "random_state":          42,
            "early_stopping_rounds": 200,
        },
        "catboost": {
            "iterations":            5000,
            "learning_rate":         0.01,
            "depth":                 6,
            "l2_leaf_reg":           3,
            "border_count":          128,
            "random_seed":           42,
            "early_stopping_rounds": 200,
        },
        "nn": {
            "hidden_dims":  [512, 256, 128, 64],
            "dropout":      0.3,
            "lr":           1e-3,
            "weight_decay": 1e-4,
            "epochs":       300,
            "batch_size":   64,
            "patience":     30,
        },
        "stacking": {
            "meta_alphas": [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0],
        },
    },
}

py_config = OmegaConf.create(config)
