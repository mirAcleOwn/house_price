from pathlib import Path

from omegaconf import OmegaConf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "House Prices Regression Techniques"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SEED = 42


config = {
    "seed": SEED,
    "paths": {
        "train": str(DATA_DIR / "train.csv"),
        "test": str(DATA_DIR / "test.csv"),
        "artifacts_dir": str(ARTIFACTS_DIR),
    },
    "data": {
        "target_col": "SalePrice",
        "id_col": "Id",
        "target_transform": "log1p",
    },
    "validation": {
        "stratify_bins": 10,
        "single_split_valid_size": 0.2,
        "shuffle": True,
        "compare_n_splits": [5],
        "primary_n_splits": 5,
    },
    "eda": {
        "enabled": True,
        "plots_dir": str(ARTIFACTS_DIR / "eda"),
        "top_numeric_features": 15,
    },
    "preprocessing": {
        "scale_numeric_for_linear": True,
        "scale_numeric_for_knn": True,
        "numeric_imputer_strategy": "median",
        "categorical_imputer_strategy": "most_frequent",
        "drop_constant_features": True,
        "constant_threshold_unique": 1,
        "drop_correlated_features": True,
        "correlation_threshold": 0.995,
        "clip_outliers": False,
        "outlier_lower_quantile": 0.01,
        "outlier_upper_quantile": 0.99,
    },
    "feature_engineering": {
        "enabled": True,
        "add_time_features": True,
        "add_total_area": True,
        "add_total_bathrooms": True,
        "add_house_age": True,
        "add_interactions": True,
        "interaction_pairs": [
            ["OverallQual", "GrLivArea"],
            ["GarageCars", "GarageArea"],
            ["TotalBsmtSF", "1stFlrSF"],
        ],
    },
    "models": {
        "ridge": {
            "enabled": True,
            "grids": [
                {"alpha": 1.0},
                {"alpha": 3.0},
                {"alpha": 10.0},
            ],
        },
        "lasso": {
            "enabled": True,
            "grids": [
                {"alpha": 0.0005, "max_iter": 10000},
                {"alpha": 0.001, "max_iter": 10000},
            ],
        },
        "elasticnet": {
            "enabled": True,
            "grids": [
                {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10000},
            ],
        },
        "random_forest": {
            "enabled": True,
            "grids": [
                {
                    "n_estimators": 600,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "random_state": SEED,
                    "n_jobs": -1,
                }
            ],
        },
        "xgboost": {
            "enabled": False,
            "grids": [
                {
                    "n_estimators": 1500,
                    "learning_rate": 0.03,
                    "max_depth": 4,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": SEED,
                    "n_jobs": -1,
                }
            ],
        },
        "lightgbm": {
            "enabled": False,
            "grids": [
                {
                    "n_estimators": 1500,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": SEED,
                    "n_jobs": -1,
                }
            ],
        },
        "catboost": {
            "enabled": False,
            "grids": [
                {
                    "iterations": 1500,
                    "learning_rate": 0.03,
                    "depth": 6,
                    "random_seed": SEED,
                    "verbose": False,
                }
            ],
        },
        "dnn": {
            "enabled": False,
            "grids": [
                {
                    "hidden_dims": [256, 128],
                    "dropout": 0.1,
                    "epochs": 50,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "random_state": SEED,
                }
            ],
        },
    },
    "ensembles": {
        "enabled": True,
        "min_models": 2,
        "average": True,
        "voting": True,
        "stacking": True,
        "stacker_models": [
            {"name": "ridge", "params": {"alpha": 1.0}},
        ],
    },
    "output": {
        "save_submission": True,
        "submission_path": str(ARTIFACTS_DIR / "submission.csv"),
        "results_csv": str(ARTIFACTS_DIR / "results.csv"),
        "oof_dir": str(ARTIFACTS_DIR / "oof"),
        "predictions_dir": str(ARTIFACTS_DIR / "predictions"),
    },
}


py_config = OmegaConf.create(config)
