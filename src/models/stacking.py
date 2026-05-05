"""
Стекинг второго уровня: Ridge мета-леарнер.

Идея:
  1. Каждая базовая модель (Ridge, Lasso, ElasticNet, LightGBM, XGBoost, CatBoost, NN)
     даёт OOF-предсказания на обучающей выборке.
  2. Эти предсказания склеиваются в матрицу мета-признаков (n_train × n_models).
  3. Ridge обучается на этой матрице — он подбирает оптимальные веса для каждой модели,
     автоматически учитывая корреляции между ними.
  4. На тесте финальное предсказание = Ridge.predict(test_meta_features).

Почему Ridge, а не что-то сложнее:
  - Мета-признаков мало (= число базовых моделей), переобучить их трудно
  - Ridge с RobustScaler даёт стабильные веса, не зависящие от масштаба предсказаний
  - Интерпретируемые коэффициенты: сразу видно вклад каждой модели
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.utils.helpers import rmsle


def build_meta_features(
    oof_dict: dict, test_preds_dict: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Формирует матрицы мета-признаков из OOF и тестовых предсказаний.

    Параметры:
        oof_dict        : {"model_name": oof_array, ...}
        test_preds_dict : {"model_name": test_pred_array, ...}

    Возвращает:
        (meta_train, meta_test) — DataFrame с именами моделей в колонках
    """
    names      = list(oof_dict.keys())
    meta_train = pd.DataFrame({name: oof_dict[name]        for name in names})
    meta_test  = pd.DataFrame({name: test_preds_dict[name] for name in names})
    return meta_train, meta_test


def train_stacking(
    oof_dict: dict,
    test_preds_dict: dict,
    y_train,
    meta_alphas=(0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0),
) -> tuple:
    """
    Обучает Ridge мета-леарнер на OOF предсказаниях базовых моделей.

    Параметры:
        oof_dict        : OOF предсказания каждой модели на train
        test_preds_dict : предсказания каждой модели на test
        y_train         : истинные значения (log1p-пространство)
        meta_alphas     : alphas для подбора регуляризации в RidgeCV

    Возвращает:
        meta_model       : обученный Pipeline (RobustScaler + RidgeCV)
        stacked_test     : np.ndarray, финальные предсказания на test
        stack_cv_score   : float, OOF RMSLE стека
    """
    meta_train, meta_test = build_meta_features(oof_dict, test_preds_dict)

    meta_model = make_pipeline(
        RobustScaler(),
        RidgeCV(alphas=meta_alphas, cv=5),
    )
    meta_model.fit(meta_train, y_train)

    stacked_oof  = meta_model.predict(meta_train)
    stacked_test = meta_model.predict(meta_test)
    score        = rmsle(y_train, stacked_oof)

    # Показываем веса — это полезно: видно, кто вносит больший вклад
    ridge      = meta_model.named_steps["ridgecv"]
    coef_df    = pd.Series(ridge.coef_, index=meta_train.columns)
    coef_df    = coef_df.sort_values(ascending=False)

    print(f"  Stacking OOF RMSLE : {score:.5f}")
    print(f"  Best alpha         : {ridge.alpha_:.4f}")
    print("  Meta-learner weights:")
    for name, w in coef_df.items():
        print(f"    {name:<18} {w:+.4f}")

    return meta_model, stacked_test, score
