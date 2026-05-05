"""
Gradient boosting модели: LightGBM, XGBoost, CatBoost.

Каждая модель обучается с K-fold CV + early stopping:
  - early stopping предотвращает переобучение без подбора n_estimators вручную
  - OOF предсказания нужны для стекинга (честная оценка без утечки)
  - финальные предсказания на test — среднее по всем fold-моделям (ансамбль)

Импорты библиотек защищены try/except: если пакет не установлен,
функция выбросит понятную ошибку только при вызове, а не при импорте.
"""

import numpy as np
from sklearn.model_selection import KFold

from src.utils.helpers import rmsle


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lgbm(X, y, params: dict, n_folds: int = 5, random_state: int = 42) -> tuple:
    """
    LightGBM с K-fold CV и early stopping.

    LightGBM требует передать eval_set прямо в .fit(), поэтому мы не можем
    использовать generic cross_val_predict из base.py — здесь свой loop.

    Параметры:
        params   : гиперпараметры из конфига (early_stopping_rounds выделяется отдельно)

    Возвращает:
        (oof_preds, cv_score, fold_models)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    p     = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early = params.get("early_stopping_rounds", 100)

    oof    = np.zeros(len(X))
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[val_idx] = model.predict(X_val)

        best = model.best_iteration_ or p.get("n_estimators", "?")
        models.append(model)
        print(f"  Fold {fold + 1}/{n_folds}: {rmsle(y_val, oof[val_idx]):.5f}"
              f"  (trees: {best})")

    cv = rmsle(y, oof)
    print(f"  LightGBM CV: {cv:.5f}")
    return oof, cv, models


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgb(X, y, params: dict, n_folds: int = 5, random_state: int = 42) -> tuple:
    """
    XGBoost с K-fold CV и early stopping.

    Параметры:
        params : гиперпараметры из конфига (early_stopping_rounds выделяется отдельно)

    Возвращает:
        (oof_preds, cv_score, fold_models)
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("pip install xgboost")

    p     = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early = params.get("early_stopping_rounds", 100)

    oof    = np.zeros(len(X))
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            **p,
            early_stopping_rounds=early,
            eval_metric="rmse",
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx] = model.predict(X_val)

        models.append(model)
        print(f"  Fold {fold + 1}/{n_folds}: {rmsle(y_val, oof[val_idx]):.5f}"
              f"  (trees: {model.best_iteration})")

    cv = rmsle(y, oof)
    print(f"  XGBoost CV: {cv:.5f}")
    return oof, cv, models


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def train_catboost(X, y, params: dict, n_folds: int = 5, random_state: int = 42) -> tuple:
    """
    CatBoost с K-fold CV и early stopping.

    CatBoost хорошо работает с категориальными признаками напрямую,
    но здесь мы уже всё закодировали — поэтому он работает как обычный бустинг.
    Преимущество: меньше переобучения по сравнению с XGB/LGBM на малых данных.

    Параметры:
        params : гиперпараметры из конфига (early_stopping_rounds выделяется отдельно)

    Возвращает:
        (oof_preds, cv_score, fold_models)
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("pip install catboost")

    p     = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early = params.get("early_stopping_rounds", 100)

    oof    = np.zeros(len(X))
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**p, early_stopping_rounds=early)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        oof[val_idx] = model.predict(X_val)

        models.append(model)
        print(f"  Fold {fold + 1}/{n_folds}: {rmsle(y_val, oof[val_idx]):.5f}"
              f"  (trees: {model.best_iteration_})")

    cv = rmsle(y, oof)
    print(f"  CatBoost CV: {cv:.5f}")
    return oof, cv, models


# ---------------------------------------------------------------------------
# Общий predict для ансамбля фолдов
# ---------------------------------------------------------------------------

def predict_boosting(models: list, X) -> np.ndarray:
    """Усредняет предсказания всех fold-моделей одного бустинга."""
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1)


# ---------------------------------------------------------------------------
# Обёртка: обучить все три
# ---------------------------------------------------------------------------

def train_all_boosting(X, y, cfg, n_folds: int = 5, random_state: int = 42) -> dict:
    """
    Запускает LightGBM, XGBoost и CatBoost последовательно.

    Возвращает словарь вида:
        {
            "lgbm":     {"oof": np.ndarray, "cv": float, "models": list},
            "xgb":      {...},
            "catboost": {...},
        }
    """
    from omegaconf import OmegaConf

    results = {}

    print("─── LightGBM ─────────────────────────")
    lgbm_params = OmegaConf.to_container(cfg.models.lgbm, resolve=True)
    oof, cv, models = train_lgbm(X, y, lgbm_params, n_folds=n_folds, random_state=random_state)
    results["lgbm"] = {"oof": oof, "cv": cv, "models": models}

    print("─── XGBoost ──────────────────────────")
    xgb_params = OmegaConf.to_container(cfg.models.xgb, resolve=True)
    oof, cv, models = train_xgb(X, y, xgb_params, n_folds=n_folds, random_state=random_state)
    results["xgb"] = {"oof": oof, "cv": cv, "models": models}

    print("─── CatBoost ─────────────────────────")
    cat_params = OmegaConf.to_container(cfg.models.catboost, resolve=True)
    oof, cv, models = train_catboost(X, y, cat_params, n_folds=n_folds, random_state=random_state)
    results["catboost"] = {"oof": oof, "cv": cv, "models": models}

    return results
