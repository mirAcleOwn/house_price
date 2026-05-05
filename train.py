"""
Главный скрипт обучения — запускает полный пайплайн:

    Preprocessing → Linear models → Gradient boosting → (Neural Net) → Stacking

Запуск:
    python train.py

Результаты:
    - модели сохраняются в  models/
    - submission            data/submissions/submission_stacking.csv
    - таблица сравнения     выводится в консоль
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import py_config
from src.data_processing.preprocess import run_pipeline
from src.models.gradient_boosting import predict_boosting, train_all_boosting
from src.models.linear import predict_linear, train_linear_models
from src.models.stacking import train_stacking
from src.utils.artifacts import save_cv_scores, save_feature_importances, save_oof_predictions
from src.utils.helpers import Timer, rmsle, save_pickle, set_seed


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _print_leaderboard(scores: dict) -> None:
    """Таблица сравнения CV-скоров всех моделей, отсортированная по качеству."""
    print("\n" + "═" * 44)
    print(f"  {'Model':<20} {'CV RMSLE':>10}")
    print("─" * 44)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    for name, score in sorted_scores:
        marker = " ← best" if name == sorted_scores[0][0] else ""
        print(f"  {name:<20} {score:>10.5f}{marker}")
    print("═" * 44)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = py_config
    set_seed(cfg.cv.random_state)

    Path(cfg.path.submissions).mkdir(parents=True, exist_ok=True)
    Path(cfg.path.models).mkdir(parents=True, exist_ok=True)

    n_folds = cfg.cv.n_folds
    seed    = cfg.cv.random_state

    # ── Preprocessing ──────────────────────────────────────────────────────
    print("\n=== Preprocessing ===")
    with Timer("preprocessing"):
        X_train, y_train, X_test, test_ids = run_pipeline(cfg)

    oof_preds  = {}
    test_preds = {}
    cv_scores  = {}

    # ── Линейные модели ────────────────────────────────────────────────────
    print("\n=== Linear models ===")
    with Timer("linear"):
        linear_results = train_linear_models(
            X_train, y_train, cfg, n_folds=n_folds, random_state=seed,
        )
    for name, data in linear_results.items():
        oof_preds[name]  = data["oof"]
        test_preds[name] = predict_linear(data["models"], X_test)
        cv_scores[name]  = data["cv"]
        save_pickle(data["models"], Path(cfg.path.models) / f"{name}_models.pkl")

    # ── Градиентный бустинг ────────────────────────────────────────────────
    print("\n=== Gradient boosting ===")
    with Timer("boosting"):
        boost_results = train_all_boosting(
            X_train, y_train, cfg, n_folds=n_folds, random_state=seed,
        )
    for name, data in boost_results.items():
        oof_preds[name]  = data["oof"]
        test_preds[name] = predict_boosting(data["models"], X_test)
        cv_scores[name]  = data["cv"]
        save_pickle(data["models"], Path(cfg.path.models) / f"{name}_models.pkl")

    # ── Нейронная сеть (опционально — нужен torch) ─────────────────────────
    try:
        from src.models.NN_with_some_layers import predict_nn, train_nn

        print("\n=== Neural Network ===")
        with Timer("nn"):
            oof_nn, cv_nn, nn_models, nn_scalers = train_nn(
                X_train, y_train, cfg.models.nn, n_folds=n_folds, random_state=seed,
            )
        oof_preds["nn"]  = oof_nn
        test_preds["nn"] = predict_nn(nn_models, nn_scalers, X_test, cfg.models.nn)
        cv_scores["nn"]  = cv_nn
        save_pickle((nn_models, nn_scalers), Path(cfg.path.models) / "nn_models.pkl")
    except ImportError:
        print("\n[skip] torch не установлен, нейронная сеть пропускается")

    # ── Стекинг ────────────────────────────────────────────────────────────
    print("\n=== Stacking (Ridge meta-learner) ===")
    meta_alphas = tuple(cfg.models.stacking.meta_alphas)
    with Timer("stacking"):
        meta_model, stacked_test, stack_cv = train_stacking(
            oof_preds, test_preds, y_train, meta_alphas=meta_alphas,
        )
    cv_scores["stacking"] = stack_cv

    models_dir = Path(cfg.path.models)

    # Сохраняем мета-модель и порядок моделей — нужны для predict.py
    save_pickle(meta_model,            models_dir / "stacking_meta.pkl")
    save_pickle(list(oof_preds.keys()), models_dir / "model_names.pkl")

    # ── Артефакты ──────────────────────────────────────────────────────────
    print("\n=== Saving artifacts ===")
    save_oof_predictions(oof_preds, y_train, models_dir)
    save_cv_scores(cv_scores, models_dir)
    save_feature_importances(boost_results, X_train.columns.tolist(), models_dir)

    # ── Итоги ──────────────────────────────────────────────────────────────
    _print_leaderboard(cv_scores)

    # ── Submission ─────────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "Id":        test_ids.values,
        "SalePrice": np.expm1(stacked_test),
    })
    out_path = Path(cfg.path.submissions) / "submission_stacking.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(submission["SalePrice"].describe().round(0).to_string())


if __name__ == "__main__":
    main()
