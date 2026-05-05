"""
Инференс — генерация submission из уже обученных моделей.

Не переобучает модели. Загружает сохранённые артефакты из models/ и
прогоняет тестовые данные через тот же preprocessing-пайплайн.

Требует предварительного запуска train.py.

Запуск:
    python predict.py
    python predict.py --submission my_submission.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import py_config
from src.data_processing.preprocess import run_pipeline
from src.models.gradient_boosting import predict_boosting
from src.models.linear import predict_linear
from src.utils.helpers import load_pickle


def _load_models(models_dir: Path) -> dict:
    """
    Загружает все сохранённые fold-модели из models/.

    Возвращает словарь вида {"lgbm": [модели], "ridge": [модели], ...}
    плюс "stacking_meta" и "model_names".
    """
    artifacts = {}

    model_files = {
        "ridge":      "ridge_models.pkl",
        "lasso":      "lasso_models.pkl",
        "elasticnet": "elasticnet_models.pkl",
        "lgbm":       "lgbm_models.pkl",
        "xgb":        "xgb_models.pkl",
        "catboost":   "catboost_models.pkl",
        "nn":         "nn_models.pkl",
    }

    for name, fname in model_files.items():
        path = models_dir / fname
        if path.exists():
            artifacts[name] = load_pickle(path)

    artifacts["stacking_meta"] = load_pickle(models_dir / "stacking_meta.pkl")
    artifacts["model_names"]   = load_pickle(models_dir / "model_names.pkl")

    loaded = [k for k in artifacts if k not in ("stacking_meta", "model_names")]
    print(f"  Loaded models: {loaded}")
    return artifacts


def _predict_all(artifacts: dict, X_test) -> dict:
    """
    Генерирует предсказания каждой базовой модели на тестовых данных.
    Порядок и набор моделей берётся из сохранённого model_names.
    """
    test_preds = {}

    for name in artifacts["model_names"]:
        if name not in artifacts:
            raise RuntimeError(
                f"Model '{name}' listed in model_names but not found in models/. "
                "Run train.py first."
            )
        models = artifacts[name]

        if name in ("lgbm", "xgb", "catboost"):
            test_preds[name] = predict_boosting(models, X_test)
        elif name == "nn":
            # nn хранится как (models, scalers)
            from src.models.NN_with_some_layers import predict_nn
            nn_models, nn_scalers = models
            test_preds[name] = predict_nn(nn_models, nn_scalers, X_test, py_config.models.nn)
        else:
            test_preds[name] = predict_linear(models, X_test)

    return test_preds


def main(submission_name: str = "submission_predict.csv") -> None:
    cfg = py_config

    models_dir      = Path(cfg.path.models)
    submissions_dir = Path(cfg.path.submissions)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        raise RuntimeError("models/ not found. Run train.py first.")

    # ── Preprocessing ──────────────────────────────────────────────────────
    # Запускаем тот же пайплайн что и при обучении — он детерминированный,
    # поэтому X_test будет идентичен тому, что видели модели при тренировке.
    print("=== Preprocessing ===")
    _, _, X_test, test_ids = run_pipeline(cfg)

    # ── Загрузка моделей ───────────────────────────────────────────────────
    print("\n=== Loading models ===")
    artifacts = _load_models(models_dir)

    # ── Предсказания базовых моделей ───────────────────────────────────────
    print("\n=== Predicting ===")
    test_preds = _predict_all(artifacts, X_test)

    # ── Стекинг ────────────────────────────────────────────────────────────
    meta_model    = artifacts["stacking_meta"]
    meta_features = pd.DataFrame(test_preds)
    final_preds   = meta_model.predict(meta_features)

    # ── Submission ─────────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "Id":        test_ids.values,
        "SalePrice": np.expm1(final_preds),
    })
    out_path = submissions_dir / submission_name
    submission.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path}")
    print(submission["SalePrice"].describe().round(0).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission from saved models")
    parser.add_argument(
        "--submission",
        default="submission_predict.csv",
        help="Output filename inside data/submissions/",
    )
    args = parser.parse_args()
    main(submission_name=args.submission)
