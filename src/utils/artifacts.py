"""
Сохранение артефактов после обучения.

Все файлы пишутся в models/:
    oof_predictions.csv       — OOF предсказания каждой модели + y_true
    cv_scores.json            — CV RMSLE всех моделей (удобно для сравнения запусков)
    feature_importances.csv   — усреднённая важность признаков по бустинг-моделям

Почему отдельный модуль, а не просто строки в train.py:
    - train.py и так достаточно длинный
    - эти функции легко тестировать изолированно
    - при добавлении новых артефактов правки только здесь
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_oof_predictions(
    oof_dict: dict,
    y_true,
    out_dir: str | Path,
) -> Path:
    """
    Сохраняет OOF предсказания всех моделей в один CSV.

    Колонки: y_true, <model_name>, ...
    Строки: индексы обучающей выборки (0..n_train-1)

    Удобно для пост-анализа: можно открыть в ноутбуке и посмотреть,
    на каких объектах конкретная модель ошибается сильнее всего.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"y_true": np.asarray(y_true)})
    for name, preds in oof_dict.items():
        df[name] = np.asarray(preds)

    # Добавляем residuals для каждой модели — сразу видно систематические ошибки
    for name in oof_dict:
        df[f"{name}_residual"] = df["y_true"] - df[name]

    path = out_dir / "oof_predictions.csv"
    df.to_csv(path, index_label="idx")
    print(f"  OOF predictions  → {path}")
    return path


def save_cv_scores(scores: dict, out_dir: str | Path) -> Path:
    """
    Сохраняет CV RMSLE всех моделей в JSON.

    JSON выбран вместо CSV потому что его легко читать глазами и
    легко парсить в скриптах сравнения запусков.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Сортируем по скору, чтобы лучшая модель была первой
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1]))

    path = out_dir / "cv_scores.json"
    with open(path, "w") as f:
        json.dump(sorted_scores, f, indent=2)
    print(f"  CV scores        → {path}")
    return path


def save_feature_importances(
    boost_results: dict,
    feature_names: list,
    out_dir: str | Path,
) -> Path:
    """
    Сохраняет усреднённую важность признаков из LightGBM, XGBoost, CatBoost.

    Усреднение по фолдам и по моделям — более стабильная оценка чем
    важность одной модели. Итоговый столбец mean_importance отсортирован
    по убыванию для удобного просмотра в ноутбуке.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"feature": feature_names})

    for model_name, data in boost_results.items():
        models = data["models"]
        importances = _get_importances(models, model_name, len(feature_names))
        if importances is not None:
            df[f"{model_name}_importance"] = importances

    importance_cols = [c for c in df.columns if c.endswith("_importance")]
    if importance_cols:
        df["mean_importance"] = df[importance_cols].mean(axis=1)
        df = df.sort_values("mean_importance", ascending=False).reset_index(drop=True)

    path = out_dir / "feature_importances.csv"
    df.to_csv(path, index=False)
    print(f"  Feature importances → {path}")
    return path


def _get_importances(models: list, model_name: str, n_features: int):
    """Извлекает важность признаков из fold-моделей и усредняет их."""
    try:
        if model_name == "lgbm":
            arrays = [m.feature_importances_ for m in models]
        elif model_name == "xgb":
            arrays = [m.feature_importances_ for m in models]
        elif model_name == "catboost":
            arrays = [m.get_feature_importance() for m in models]
        else:
            return None

        stacked = np.vstack(arrays)
        avg = stacked.mean(axis=0)

        # Нормализуем в [0, 1] для сопоставимости между моделями
        total = avg.sum()
        return avg / total if total > 0 else avg

    except Exception:
        return None