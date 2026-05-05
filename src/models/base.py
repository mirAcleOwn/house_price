"""
Общий K-fold CV цикл для sklearn-совместимых моделей.

Выделен в отдельный модуль, чтобы не дублировать один и тот же fold-loop
в linear.py, gradient_boosting.py и т.д.

cross_val_predict принимает model_factory — callable() без аргументов,
возвращающий новый экземпляр модели. Это гарантирует, что каждый fold
получает чистую модель без утечки состояния между фолдами.
"""

import numpy as np
from sklearn.model_selection import KFold

from src.utils.helpers import rmsle


def cross_val_predict(
    model_factory,
    X,
    y,
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    K-fold кросс-валидация с сохранением out-of-fold предсказаний.

    Параметры:
        model_factory  : callable() → sklearn-совместимая модель
        X              : pd.DataFrame, признаки
        y              : pd.Series, целевая переменная (log-пространство)
        n_folds        : число фолдов
        random_state   : seed для воспроизводимости
        verbose        : выводить ли скор каждого фолда

    Возвращает:
        oof_preds  : np.ndarray, out-of-fold предсказания (len == len(X))
        cv_score   : float, итоговый RMSLE по всем OOF
        models     : list, обученные модели каждого фолда
    """
    oof    = np.zeros(len(X))
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    models = []
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

        model = model_factory()
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)

        fold_score = rmsle(y_val, oof[val_idx])
        scores.append(fold_score)
        models.append(model)

        if verbose:
            print(f"  Fold {fold + 1}/{n_folds}: {fold_score:.5f}")

    cv_score = rmsle(y, oof)
    if verbose:
        print(f"  CV: {cv_score:.5f}  (std {np.std(scores):.5f})")

    return oof, cv_score, models
