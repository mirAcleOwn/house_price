from __future__ import annotations

import copy
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from config import py_config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config() -> Dict[str, Any]:
    if isinstance(py_config, dict):
        return copy.deepcopy(py_config)
    return OmegaConf.to_container(py_config, resolve=False)


def build_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def transform_target(y: pd.Series, mode: str) -> np.ndarray:
    values = y.to_numpy(dtype=float)
    if mode == "log1p":
        return np.log1p(values)
    return values


def inverse_target(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(values)
    return values


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_stratify_labels(y: pd.Series, bins: int) -> np.ndarray:
    n_bins = min(max(2, bins), max(2, y.nunique()))
    labels = pd.qcut(y.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    return labels.to_numpy()


def build_splits(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    stratify_labels = make_stratify_labels(y, cfg["validation"]["stratify_bins"])
    if n_splits <= 1:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=cfg["validation"]["single_split_valid_size"],
            random_state=cfg["seed"],
        )
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=cfg["validation"]["shuffle"],
            random_state=cfg["seed"],
        )
    return list(splitter.split(X, stratify_labels))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not self.config.get("enabled", True):
            return df

        if self.config.get("add_time_features", True):
            if {"YrSold", "MoSold"}.issubset(df.columns):
                df["SaleMonthIndex"] = df["YrSold"] * 12 + df["MoSold"]
            if "YrSold" in df.columns:
                df["YearsSinceSale2010"] = 2010 - df["YrSold"]

        if self.config.get("add_total_area", True):
            area_parts = [c for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea"] if c in df.columns]
            if area_parts:
                df["TotalUsableArea"] = df[area_parts].fillna(0).sum(axis=1)

        if self.config.get("add_total_bathrooms", True):
            bath_map = {
                "FullBath": 1.0,
                "HalfBath": 0.5,
                "BsmtFullBath": 1.0,
                "BsmtHalfBath": 0.5,
            }
            cols = [c for c in bath_map if c in df.columns]
            if cols:
                df["TotalBathrooms"] = sum(df[c].fillna(0) * bath_map[c] for c in cols)

        if self.config.get("add_house_age", True):
            if {"YrSold", "YearBuilt"}.issubset(df.columns):
                df["HouseAgeAtSale"] = df["YrSold"] - df["YearBuilt"]
            if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
                df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

        if self.config.get("add_interactions", True):
            for left, right in self.config.get("interaction_pairs", []):
                if left in df.columns and right in df.columns:
                    df[f"{left}_x_{right}"] = df[left].fillna(0) * df[right].fillna(0)

        return df


class DataFrameFeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        drop_constant_features: bool = True,
        constant_threshold_unique: int = 1,
        drop_correlated_features: bool = True,
        correlation_threshold: float = 0.98,
    ):
        self.drop_constant_features = drop_constant_features
        self.constant_threshold_unique = constant_threshold_unique
        self.drop_correlated_features = drop_correlated_features
        self.correlation_threshold = correlation_threshold
        self.columns_to_drop_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataFrameFeatureCleaner":
        df = X.copy()
        drop_cols: List[str] = []

        if self.drop_constant_features:
            for column in df.columns:
                if df[column].nunique(dropna=False) <= self.constant_threshold_unique:
                    drop_cols.append(column)

        if self.drop_correlated_features:
            numeric_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr = numeric_df.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                for column in upper.columns:
                    if (upper[column] > self.correlation_threshold).any():
                        drop_cols.append(column)

        self.columns_to_drop_ = sorted(set(drop_cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop_, errors="ignore").copy()


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, enabled: bool = True, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.enabled = enabled
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_: pd.Series | None = None
        self.upper_bounds_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OutlierClipper":
        if not self.enabled:
            return self
        numeric_df = X.select_dtypes(include=[np.number])
        self.lower_bounds_ = numeric_df.quantile(self.lower_quantile)
        self.upper_bounds_ = numeric_df.quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or self.lower_bounds_ is None or self.upper_bounds_ is None:
            return X.copy()
        df = X.copy()
        numeric_cols = list(self.lower_bounds_.index)
        df[numeric_cols] = df[numeric_cols].clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return df


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dims: Iterable[int] = (256, 128),
        activation: str = "relu",
        batch_norm: bool = False,
        dropout: float = 0.0,
        optimizer: str = "adam",
        scheduler: str | None = None,
        lr: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        weight_decay: float = 0.0,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state
        self.model_: nn.Module | None = None

    def _activation(self) -> nn.Module:
        mapping = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return mapping.get(self.activation, nn.ReLU())

    def _build_model(self, input_dim: int) -> nn.Module:
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._activation())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        return nn.Sequential(*layers)

    def _build_optimizer(self, model: nn.Module):
        if self.optimizer == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        return torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPRegressor":
        if torch is None:
            raise ImportError("PyTorch is not installed, but the DNN model is enabled.")

        torch.manual_seed(self.random_state)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_samples = len(X_np)
        if n_samples < 10:
            raise ValueError("Not enough samples for DNN training.")

        split_idx = max(1, int(n_samples * 0.9))
        indices = np.arange(n_samples)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)
        train_idx = indices[:split_idx]
        valid_idx = indices[split_idx:] if split_idx < n_samples else indices[-max(1, n_samples // 10):]

        train_ds = TensorDataset(torch.from_numpy(X_np[train_idx]), torch.from_numpy(y_np[train_idx]))
        valid_x = torch.from_numpy(X_np[valid_idx])
        valid_y = torch.from_numpy(y_np[valid_idx])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        model = self._build_model(X_np.shape[1])
        optimizer = self._build_optimizer(model)
        criterion = nn.MSELoss()

        scheduler = None
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_state = None
        best_loss = float("inf")
        epochs_without_improvement = 0

        for _ in range(self.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_pred = model(valid_x)
                valid_loss = criterion(valid_pred, valid_y).item()

            if scheduler is not None:
                scheduler.step()

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model is not fitted.")
        X_np = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X_np)).numpy().reshape(-1)
        return pred


@dataclass
class ExperimentResult:
    experiment_name: str
    model_name: str
    params: Dict[str, Any]
    primary_score: float
    fold_scores: Dict[int, float]
    oof_predictions: np.ndarray | None
    fitted_model: Any | None = None
    test_predictions: np.ndarray | None = None


def run_eda(train_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    if not cfg["eda"]["enabled"]:
        return

    eda_dir = ensure_dir(Path(cfg["eda"]["plots_dir"]))
    target_col = cfg["data"]["target_col"]

    stats = train_df.describe(include="all").transpose()
    stats.to_csv(eda_dir / "train_summary_stats.csv")

    missing = train_df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing.to_csv(eda_dir / "missing_values.csv", header=["missing_count"])

    plt.figure(figsize=(8, 5))
    train_df[target_col].hist(bins=40)
    plt.title("Target Distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(eda_dir / "target_distribution.png")
    plt.close()

    if not missing.empty:
        plt.figure(figsize=(12, 6))
        missing.head(20).plot(kind="bar")
        plt.title("Top Missing Features")
        plt.ylabel("Missing Count")
        plt.tight_layout()
        plt.savefig(eda_dir / "missing_top20.png")
        plt.close()

    numeric_features = train_df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    if not numeric_features.empty:
        correlations = numeric_features.corrwith(train_df[target_col]).abs().sort_values(ascending=False)
        top_features = correlations.head(cfg["eda"]["top_numeric_features"]).index.tolist()
        for feature in top_features:
            plt.figure(figsize=(7, 5))
            plt.scatter(train_df[feature], train_df[target_col], alpha=0.4)
            plt.title(f"{feature} vs {target_col}")
            plt.xlabel(feature)
            plt.ylabel(target_col)
            plt.tight_layout()
            plt.savefig(eda_dir / f"scatter_{feature}.png")
            plt.close()


def get_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(model_name: str, X: pd.DataFrame, cfg: Dict[str, Any]) -> ColumnTransformer:
    numeric_cols, categorical_cols = get_feature_columns(X)

    scale_numeric = False
    if model_name in {"linear_regression", "ridge", "lasso", "elasticnet"}:
        scale_numeric = cfg["preprocessing"]["scale_numeric_for_linear"]
    if model_name == "knn":
        scale_numeric = cfg["preprocessing"]["scale_numeric_for_knn"]
    if model_name == "dnn":
        scale_numeric = True

    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=cfg["preprocessing"]["numeric_imputer_strategy"]))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=cfg["preprocessing"]["categorical_imputer_strategy"])),
        ("encoder", build_onehot_encoder()),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ],
        remainder="drop",
    )


def build_estimator(model_name: str, params: Dict[str, Any]) -> Any:
    if model_name == "linear_regression":
        return LinearRegression(**params)
    if model_name == "ridge":
        return Ridge(**params)
    if model_name == "lasso":
        return Lasso(**params)
    if model_name == "elasticnet":
        return ElasticNet(**params)
    if model_name == "knn":
        return KNeighborsRegressor(**params)
    if model_name == "decision_tree":
        return DecisionTreeRegressor(**params)
    if model_name == "random_forest":
        return RandomForestRegressor(**params)
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        xgb_params = {"objective": "reg:squarederror", **params}
        return XGBRegressor(**xgb_params)
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        return LGBMRegressor(**params)
    if model_name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        return CatBoostRegressor(**params)
    if model_name == "dnn":
        return TorchMLPRegressor(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def build_model_pipeline(model_name: str, params: Dict[str, Any], X_sample: pd.DataFrame, cfg: Dict[str, Any]) -> Any:
    preprocess_cfg = cfg["preprocessing"]
    cleaner = DataFrameFeatureCleaner(
        drop_constant_features=preprocess_cfg["drop_constant_features"],
        constant_threshold_unique=preprocess_cfg["constant_threshold_unique"],
        drop_correlated_features=preprocess_cfg["drop_correlated_features"],
        correlation_threshold=preprocess_cfg["correlation_threshold"],
    )
    cleaner.fit(X_sample)
    cleaned_sample = cleaner.transform(X_sample)

    clipper = OutlierClipper(
        enabled=preprocess_cfg["clip_outliers"],
        lower_quantile=preprocess_cfg["outlier_lower_quantile"],
        upper_quantile=preprocess_cfg["outlier_upper_quantile"],
    )
    clipper.fit(cleaned_sample)

    steps: List[Tuple[str, Any]] = [("cleaner", cleaner), ("outlier_clipper", clipper)]

    estimator = build_estimator(model_name, params)

    if model_name == "catboost":
        steps.append(("model", estimator))
        return Pipeline(steps)

    steps.append(("preprocessor", build_preprocessor(model_name, cleaned_sample, cfg)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_predict_catboost(pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame) -> np.ndarray:
    cleaner = pipeline.named_steps["cleaner"]
    clipper = pipeline.named_steps["outlier_clipper"]
    model = pipeline.named_steps["model"]

    X_train_prep = clipper.fit_transform(cleaner.fit_transform(X_train))
    X_valid_prep = clipper.transform(cleaner.transform(X_valid))

    numeric_cols, categorical_cols = get_feature_columns(X_train_prep)
    for column in numeric_cols:
        fill_value = X_train_prep[column].median()
        X_train_prep[column] = X_train_prep[column].fillna(fill_value)
        X_valid_prep[column] = X_valid_prep[column].fillna(fill_value)
    for column in categorical_cols:
        X_train_prep[column] = X_train_prep[column].fillna("Missing")
        X_valid_prep[column] = X_valid_prep[column].fillna("Missing")

    model.fit(X_train_prep, y_train, cat_features=categorical_cols)
    return model.predict(X_valid_prep)


def fit_full_catboost(pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    cleaner = pipeline.named_steps["cleaner"]
    clipper = pipeline.named_steps["outlier_clipper"]
    model = pipeline.named_steps["model"]

    X_train_prep = clipper.fit_transform(cleaner.fit_transform(X_train))
    X_test_prep = clipper.transform(cleaner.transform(X_test))

    numeric_cols, categorical_cols = get_feature_columns(X_train_prep)
    for column in numeric_cols:
        fill_value = X_train_prep[column].median()
        X_train_prep[column] = X_train_prep[column].fillna(fill_value)
        X_test_prep[column] = X_test_prep[column].fillna(fill_value)
    for column in categorical_cols:
        X_train_prep[column] = X_train_prep[column].fillna("Missing")
        X_test_prep[column] = X_test_prep[column].fillna("Missing")

    model.fit(X_train_prep, y_train, cat_features=categorical_cols)
    return model.predict(X_test_prep)


def evaluate_experiment(
    model_name: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
) -> ExperimentResult:
    compare_splits = cfg["validation"]["compare_n_splits"]
    primary_splits = cfg["validation"]["primary_n_splits"]
    target_mode = cfg["data"]["target_transform"]
    y_transformed = transform_target(y, target_mode)
    fold_scores: Dict[int, float] = {}
    oof_predictions: np.ndarray | None = None

    for split_count in compare_splits:
        fold_indices = build_splits(X, y, cfg, split_count)
        preds_for_split = np.full(len(X), np.nan, dtype=float) if split_count == primary_splits else None
        scores: List[float] = []

        for train_idx, valid_idx in fold_indices:
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y_transformed[train_idx]
            y_valid = y_transformed[valid_idx]

            pipeline = build_model_pipeline(model_name, params, X_train, cfg)
            if model_name == "catboost":
                valid_pred = fit_predict_catboost(pipeline, X_train, y_train, X_valid)
            else:
                pipeline.fit(X_train, y_train)
                valid_pred = pipeline.predict(X_valid)

            scores.append(rmse(y_valid, valid_pred))
            if preds_for_split is not None:
                preds_for_split[valid_idx] = valid_pred

        fold_scores[split_count] = float(np.mean(scores))
        if preds_for_split is not None:
            oof_predictions = preds_for_split

    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    experiment_name = f"{model_name}__{params_hash}"
    return ExperimentResult(
        experiment_name=experiment_name,
        model_name=model_name,
        params=params,
        primary_score=fold_scores[primary_splits],
        fold_scores=fold_scores,
        oof_predictions=oof_predictions,
    )


def fit_full_experiment(
    result: ExperimentResult,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cfg: Dict[str, Any],
) -> ExperimentResult:
    y_transformed = transform_target(y_train, cfg["data"]["target_transform"])
    pipeline = build_model_pipeline(result.model_name, result.params, X_train, cfg)

    if result.model_name == "catboost":
        test_predictions = fit_full_catboost(pipeline, X_train, y_transformed, X_test)
        result.fitted_model = pipeline
        result.test_predictions = test_predictions
        return result

    pipeline.fit(X_train, y_transformed)
    result.fitted_model = pipeline
    result.test_predictions = pipeline.predict(X_test)
    return result


def iter_model_experiments(cfg: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for model_name, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", False):
            continue
        for params in model_cfg.get("grids", []):
            yield model_name, params


def results_to_frame(results: List[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "experiment_name": result.experiment_name,
            "model_name": result.model_name,
            "primary_score": result.primary_score,
            "params": json.dumps(result.params, sort_keys=True),
        }
        for split_count, score in result.fold_scores.items():
            row[f"rmse_{split_count}_folds"] = score
        rows.append(row)
    return pd.DataFrame(rows).sort_values("primary_score").reset_index(drop=True)


def run_stacking(
    fitted_results: List[ExperimentResult],
    y_train: pd.Series,
    cfg: Dict[str, Any],
) -> List[ExperimentResult]:
    usable_results = [r for r in fitted_results if r.oof_predictions is not None and r.test_predictions is not None]
    if len(usable_results) < cfg["ensembles"]["min_models"]:
        return []

    oof_matrix = np.column_stack([r.oof_predictions for r in usable_results])
    test_matrix = np.column_stack([r.test_predictions for r in usable_results])
    y_transformed = transform_target(y_train, cfg["data"]["target_transform"])
    ensemble_results: List[ExperimentResult] = []

    if cfg["ensembles"]["average"]:
        avg_oof = oof_matrix.mean(axis=1)
        avg_test = test_matrix.mean(axis=1)
        ensemble_results.append(
            ExperimentResult(
                experiment_name="ensemble_average",
                model_name="ensemble_average",
                params={"members": [r.experiment_name for r in usable_results]},
                primary_score=rmse(y_transformed, avg_oof),
                fold_scores={cfg["validation"]["primary_n_splits"]: rmse(y_transformed, avg_oof)},
                oof_predictions=avg_oof,
                test_predictions=avg_test,
            )
        )

    if cfg["ensembles"]["voting"]:
        vote_oof = np.median(oof_matrix, axis=1)
        vote_test = np.median(test_matrix, axis=1)
        ensemble_results.append(
            ExperimentResult(
                experiment_name="ensemble_voting",
                model_name="ensemble_voting",
                params={"members": [r.experiment_name for r in usable_results]},
                primary_score=rmse(y_transformed, vote_oof),
                fold_scores={cfg["validation"]["primary_n_splits"]: rmse(y_transformed, vote_oof)},
                oof_predictions=vote_oof,
                test_predictions=vote_test,
            )
        )

    if cfg["ensembles"]["stacking"]:
        for stacker_cfg in cfg["ensembles"]["stacker_models"]:
            name = stacker_cfg["name"]
            params = stacker_cfg.get("params", {})
            estimator = build_estimator(name, params)
            estimator.fit(oof_matrix, y_transformed)
            stack_oof = estimator.predict(oof_matrix)
            stack_test = estimator.predict(test_matrix)
            ensemble_results.append(
                ExperimentResult(
                    experiment_name=f"ensemble_stacking_{name}",
                    model_name=f"ensemble_stacking_{name}",
                    params={"members": [r.experiment_name for r in usable_results], "stacker": params},
                    primary_score=rmse(y_transformed, stack_oof),
                    fold_scores={cfg["validation"]["primary_n_splits"]: rmse(y_transformed, stack_oof)},
                    oof_predictions=stack_oof,
                    test_predictions=stack_test,
                )
            )

    return ensemble_results


def save_submission(best_result: ExperimentResult, test_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    if best_result.test_predictions is None or not cfg["output"]["save_submission"]:
        return
    predictions = inverse_target(best_result.test_predictions, cfg["data"]["target_transform"])
    predictions = np.clip(predictions, a_min=0, a_max=None)
    submission = pd.DataFrame(
        {
            cfg["data"]["id_col"]: test_df[cfg["data"]["id_col"]],
            cfg["data"]["target_col"]: predictions,
        }
    )
    Path(cfg["output"]["submission_path"]).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(cfg["output"]["submission_path"], index=False)


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()

    artifacts_dir = ensure_dir(Path(cfg["paths"]["artifacts_dir"]))
    ensure_dir(Path(cfg["output"]["oof_dir"]))
    ensure_dir(Path(cfg["output"]["predictions_dir"]))

    train_df = pd.read_csv(cfg["paths"]["train"])
    test_df = pd.read_csv(cfg["paths"]["test"])

    run_eda(train_df, cfg)

    feature_engineer = FeatureEngineer(cfg["feature_engineering"])
    X_train = feature_engineer.fit_transform(train_df.drop(columns=[cfg["data"]["target_col"]]))
    X_test = feature_engineer.transform(test_df.copy())
    y_train = train_df[cfg["data"]["target_col"]].copy()

    experiment_results: List[ExperimentResult] = []
    for model_name, params in iter_model_experiments(cfg):
        try:
            result = evaluate_experiment(model_name, params, X_train, y_train, cfg)
            experiment_results.append(result)
            print(f"{result.experiment_name}: RMSE={result.primary_score:.5f}")
        except Exception as exc:
            print(f"Skipping {model_name} with params {params}: {exc}")

    if not experiment_results:
        raise RuntimeError("No experiments finished successfully.")

    fitted_results: List[ExperimentResult] = []
    for result in sorted(experiment_results, key=lambda item: item.primary_score):
        try:
            fitted_results.append(fit_full_experiment(result, X_train, y_train, X_test, cfg))
        except Exception as exc:
            print(f"Failed to fit full model for {result.experiment_name}: {exc}")

    ensemble_results: List[ExperimentResult] = []
    if cfg["ensembles"]["enabled"]:
        ensemble_results = run_stacking(fitted_results, y_train, cfg)

    results_df = results_to_frame(experiment_results + ensemble_results)
    results_df.to_csv(cfg["output"]["results_csv"], index=False)

    all_results = fitted_results + ensemble_results
    if not all_results:
        raise RuntimeError("Cross-validation completed, but no model could be fitted on full data.")
    best_result = min(all_results, key=lambda item: item.primary_score)
    save_submission(best_result, test_df, cfg)

    if best_result.oof_predictions is not None:
        oof_path = Path(cfg["output"]["oof_dir"]) / f"{best_result.experiment_name}.npy"
        np.save(oof_path, best_result.oof_predictions)
    if best_result.test_predictions is not None:
        pred_path = Path(cfg["output"]["predictions_dir"]) / f"{best_result.experiment_name}.npy"
        np.save(pred_path, best_result.test_predictions)

    summary_path = artifacts_dir / "best_result.json"
    summary_payload = {
        "experiment_name": best_result.experiment_name,
        "model_name": best_result.model_name,
        "primary_score": best_result.primary_score,
        "params": best_result.params,
        "fold_scores": best_result.fold_scores,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print(f"Best experiment: {best_result.experiment_name}")
    print(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
