"""
Вспомогательные утилиты, которые используются во всём проекте.

  set_seed   — воспроизводимость экспериментов
  rmsle      — метрика соревнования (RMSE в log-пространстве)
  Timer      — контекстный менеджер для замера времени
  save/load_pickle — сериализация моделей
"""

import pickle
import random
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Фиксирует seed для Python, NumPy и (если установлен) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rmsle(y_true, y_pred) -> float:
    """
    Root Mean Squared Log Error — официальная метрика Kaggle House Prices.

    Поскольку мы тренируем модели уже на log1p(target), это просто RMSE
    между предсказанием и истинным значением в log-пространстве.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def save_pickle(obj, path) -> None:
    """Сохраняет объект в pickle, создавая директорию при необходимости."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Timer:
    """
    Контекстный менеджер для замера времени исполнения блока.

    Использование:
        with Timer("LightGBM"):
            train_lgbm(...)
        # >> [LightGBM] 42.3s
    """

    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        label = f"[{self.name}] " if self.name else ""
        print(f"{label}{elapsed:.1f}s")
