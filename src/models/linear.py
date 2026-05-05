"""
Линейные модели: Ridge, Lasso, ElasticNet.

Все три оборачиваются в Pipeline с RobustScaler — он устойчив к выбросам
в признаках (в отличие от StandardScaler), что важно для Ames Housing где
есть сильно скошенные колонки даже после log1p.

Для подбора гиперпараметров используется встроенный CV:
  - RidgeCV    — перебирает alphas через обобщённую кросс-валидацию (GCV)
  - LassoCV    — coordinate descent + k-fold по alpha
  - ElasticNetCV — то же + поиск по l1_ratio

Внешний K-fold в cross_val_predict (base.py) даёт честные OOF предсказания
для стекинга. Получается nested CV: inner CV выбирает alpha, outer — оценивает.
"""

from omegaconf import OmegaConf
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.models.base import cross_val_predict


def _make_ridge(alphas):
    return make_pipeline(RobustScaler(), RidgeCV(alphas=alphas, cv=5))


def _make_lasso(alphas, max_iter):
    return make_pipeline(
        RobustScaler(),
        LassoCV(alphas=alphas, max_iter=max_iter, cv=5, random_state=42),
    )


def _make_elasticnet(alphas, l1_ratios, max_iter):
    return make_pipeline(
        RobustScaler(),
        ElasticNetCV(
            alphas=alphas, l1_ratio=l1_ratios,
            max_iter=max_iter, cv=5, random_state=42,
        ),
    )


def train_linear_models(X, y, cfg, n_folds: int = 5, random_state: int = 42) -> dict:
    """
    Обучает Ridge, Lasso и ElasticNet с K-fold кросс-валидацией.

    Возвращает словарь вида:
        {
            "ridge":      {"oof": np.ndarray, "cv": float, "models": list},
            "lasso":      {...},
            "elasticnet": {...},
        }
    """
    ridge_alphas = tuple(OmegaConf.to_container(cfg.models.ridge.alphas))
    lasso_alphas = tuple(OmegaConf.to_container(cfg.models.lasso.alphas))
    en_alphas    = tuple(OmegaConf.to_container(cfg.models.elasticnet.alphas))
    en_l1ratios  = list(OmegaConf.to_container(cfg.models.elasticnet.l1_ratios))

    results = {}

    print("─── Ridge ────────────────────────────")
    oof, score, models = cross_val_predict(
        lambda: _make_ridge(ridge_alphas),
        X, y, n_folds=n_folds, random_state=random_state,
    )
    results["ridge"] = {"oof": oof, "cv": score, "models": models}

    print("─── Lasso ────────────────────────────")
    oof, score, models = cross_val_predict(
        lambda: _make_lasso(lasso_alphas, cfg.models.lasso.max_iter),
        X, y, n_folds=n_folds, random_state=random_state,
    )
    results["lasso"] = {"oof": oof, "cv": score, "models": models}

    print("─── ElasticNet ───────────────────────")
    oof, score, models = cross_val_predict(
        lambda: _make_elasticnet(en_alphas, en_l1ratios, cfg.models.elasticnet.max_iter),
        X, y, n_folds=n_folds, random_state=random_state,
    )
    results["elasticnet"] = {"oof": oof, "cv": score, "models": models}

    return results


def predict_linear(models: list, X) -> object:
    """Усредняет предсказания всех CV-фолдов одной модели."""
    import numpy as np
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)
