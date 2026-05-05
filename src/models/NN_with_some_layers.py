"""
MLP для таблчных данных (Ames Housing).

Архитектура:
    Linear → BatchNorm → SiLU → Dropout  (× N слоёв)
    → Linear(1)

Детали реализации:
  - SiLU (Swish) вместо ReLU — плавнее градиенты, меньше dying neurons
  - HuberLoss вместо MSE — менее чувствителен к выбросам в y
  - Cosine LR annealing — мягкое снижение learning rate
  - Gradient clipping — стабилизирует обучение на табличных данных
  - RobustScaler — нейросеть чувствительна к масштабу признаков,
    Robust устойчивее к выбросам чем Standard
  - Early stopping с сохранением лучших весов по val loss

Для табличных данных нейронная сеть обычно уступает бустингу,
но добавляет разнообразие в ансамбль — это помогает стекингу.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils.helpers import rmsle


class MLP(nn.Module):
    """
    Многослойный перцептрон для регрессии.

    Параметры:
        input_dim   : число входных признаков
        hidden_dims : список размеров скрытых слоёв, например [512, 256, 128, 64]
        dropout     : вероятность dropout (применяется после каждого скрытого слоя)
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Kaiming инициализация — оптимальна для SiLU/ReLU-подобных активаций
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_nn(
    X,
    y,
    cfg,
    n_folds: int = 5,
    random_state: int = 42,
    device=None,
) -> tuple:
    """
    K-fold обучение MLP с early stopping.

    Параметры:
        X           : pd.DataFrame, признаки
        y           : pd.Series, log1p(SalePrice)
        cfg         : секция cfg.models.nn из конфига
        n_folds     : число фолдов
        random_state: seed
        device      : torch.device (по умолчанию cuda если доступен)

    Возвращает:
        (oof_preds, cv_score, fold_models, fold_scalers)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_arr  = X.values.astype(np.float32)
    y_arr  = y.values.astype(np.float32)
    oof    = np.zeros(len(X_arr))
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    models  = []
    scalers = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        scaler = RobustScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        train_dl = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        )

        model     = MLP(X_tr.shape[1], list(cfg.hidden_dims), cfg.dropout).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        criterion = nn.HuberLoss()

        best_loss, patience_ctr, best_weights = float("inf"), 0, None

        for epoch in range(cfg.epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(torch.tensor(X_val).to(device)).cpu().numpy()
            val_loss = np.sqrt(np.mean((y_val - val_preds) ** 2))

            if val_loss < best_loss:
                best_loss     = val_loss
                patience_ctr  = 0
                best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            oof[val_idx] = model(torch.tensor(X_val).to(device)).cpu().numpy()

        stopped_at = epoch + 1 - patience_ctr
        print(f"  Fold {fold + 1}/{n_folds}: {rmsle(y_val, oof[val_idx]):.5f}"
              f"  (epoch {stopped_at})")

        models.append(model.cpu())
        scalers.append(scaler)

    cv = rmsle(y, oof)
    print(f"  Neural Net CV: {cv:.5f}")
    return oof, cv, models, scalers


def predict_nn(models: list, scalers: list, X, cfg, device=None) -> np.ndarray:
    """
    Усредняет предсказания всех fold-моделей.

    Параметры:
        models  : список обученных MLP (по одному на фолд)
        scalers : список RobustScaler (по одному на фолд)
        X       : pd.DataFrame, тестовые признаки

    Возвращает:
        np.ndarray, усреднённые предсказания
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_arr      = X.values.astype(np.float32)
    fold_preds = []

    for model, scaler in zip(models, scalers):
        X_scaled = scaler.transform(X_arr)
        model.eval().to(device)
        with torch.no_grad():
            preds = model(torch.tensor(X_scaled).to(device)).cpu().numpy()
        fold_preds.append(preds)
        model.cpu()

    return np.mean(fold_preds, axis=0)
