"""
Оркестратор полного preprocessing-пайплайна.

Порядок:
    1. Загрузка сырых данных (load_raw_data)
    2. Удаление выбросов (только train)
    3. Импутация пропусков (fit на train, transform на train+test)
    4. Feature engineering (build_features)

Публичный интерфейс: run_pipeline(cfg) → (X_train, y_train, X_test, test_ids)
y_train уже в log1p-пространстве — inverse transform через np.expm1.
"""

import numpy as np
import pandas as pd

from src.data_processing.load import load_raw_data
from src.data_processing.impute import apply_imputer, build_imputer
from src.features.build_features import build_features


def _remove_outliers(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Удаляет два хорошо известных выброса Ames Housing:
    дома с очень большой площадью, проданные сильно ниже рынка.

    GrLivArea >= 4000 AND SalePrice < 300k — явные нерыночные сделки,
    которые смещают обучение в сторону неверных зависимостей.
    """
    mask    = ~((df["GrLivArea"] >= 4000) & (df[target] < 300_000))
    removed = (~mask).sum()
    if removed:
        print(f"  Outliers removed: {removed}")
    return df[mask].reset_index(drop=True)


def run_pipeline(cfg) -> tuple:
    """
    Полный preprocessing от сырых CSV до готовых матриц признаков.

    Параметры:
        cfg : OmegaConf-конфиг (config.py -> py_config)

    Возвращает:
        X_train  : pd.DataFrame, обучающие признаки
        y_train  : pd.Series, log1p(SalePrice)
        X_test   : pd.DataFrame, тестовые признаки (те же колонки)
        test_ids : pd.Series, оригинальные Id для submission
    """
    target = cfg.data.target
    id_col = cfg.data.id_col

    train_raw, test_raw = load_raw_data(cfg.path.raw.train, cfg.path.raw.test)

    test_ids    = test_raw[id_col].copy()
    train_clean = train_raw.drop(columns=[id_col])
    test_clean  = test_raw.drop(columns=[id_col])

    train_clean = _remove_outliers(train_clean, target)

    # Импутер обучается только на train — чтобы не было утечки из test
    imputer   = build_imputer(train_clean, target)
    train_imp = apply_imputer(imputer, train_clean, target)
    test_imp  = apply_imputer(imputer, test_clean, None)

    y_train = np.log1p(train_imp.pop(target))

    X_train, X_test = build_features(
        train_imp,
        test_imp,
        skew_threshold=cfg.features.skew_threshold,
    )

    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")
    return X_train, y_train, X_test, test_ids
