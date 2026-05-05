# Ames Housing Price Prediction

Решение соревнования [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) на Kaggle.

**Текущий результат:** 0.1256 RMSLE на публичном лидерборде.

---

## Структура репозитория

```
review_house_pricing/
│
├── config/
│   └── config.py               # Все настройки проекта: пути к данным,
│                               # гиперпараметры моделей, параметры CV
│
├── src/
│   ├── data_processing/
│   │   ├── load.py             # Загрузка CSV, быстрый обзор датасета
│   │   ├── impute.py           # Заполнение пропусков через ColumnTransformer:
│   │   │                       # качественные → 'None', числовые → 0/median
│   │   └── preprocess.py       # Оркестратор: загрузка → выбросы → импутация
│   │                           # → фичи. Возвращает готовые X_train / X_test
│   │
│   ├── features/
│   │   └── build_features.py   # Feature engineering:
│   │                           # - Ordinal encoding (24 качественных колонки)
│   │                           # - 15 доменных признаков (TotalSF, HouseAge,
│   │                           #   TotalBathrooms, QualArea и др.)
│   │                           # - Log1p для скошенных признаков
│   │                           # - One-hot encoding + выравнивание train/test
│   │
│   ├── eda/
│   │   └── plots.py            # Визуализация для ноутбука: распределение
│   │                           # таргета, пропуски, корреляции, выбросы,
│   │                           # тепловая карта корреляций
│   │
│   ├── models/
│   │   ├── base.py             # Общий K-fold CV цикл для sklearn-моделей
│   │   ├── linear.py           # Ridge, Lasso, ElasticNet + RobustScaler
│   │   ├── gradient_boosting.py# LightGBM, XGBoost, CatBoost с early stopping
│   │   ├── stacking.py         # Ridge мета-леарнер поверх OOF предсказаний
│   │   └── NN_with_some_layers.py  # MLP: BatchNorm → SiLU → Dropout,
│   │                               # HuberLoss, cosine LR, early stopping
│   │
│   └── utils/
│       ├── helpers.py          # set_seed, rmsle, Timer, save/load_pickle
│       └── artifacts.py        # Сохранение OOF предсказаний, CV скоров,
│                               # важности признаков после обучения
│
├── notebooks/
│   └── 01_EDA.ipynb            # Разведочный анализ: распределения,
│                               # пропуски, корреляции, выбросы
│
├── data/
│   ├── raw/                    # Исходные данные с Kaggle
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── data_description.txt
│   ├── processed/              # Промежуточные данные (не в git)
│   └── submissions/            # CSV-файлы для загрузки на Kaggle (не в git)
│
├── train.py                    # Запустить обучение всех моделей
└── predict.py                  # Сгенерировать submission из сохранённых моделей
```

---

## Пайплайн

```
train.csv / test.csv
      │
      ▼
Preprocessing
  • Удаление 2 выбросов (GrLivArea ≥ 4000 & SalePrice < 300k)
  • Импутация пропусков (ColumnTransformer)
  • Feature engineering: 15 новых признаков + ordinal encoding + log1p + OHE
      │
      ├──► Ridge / Lasso / ElasticNet   ──┐
      ├──► LightGBM                     ──┤  OOF predictions
      ├──► XGBoost                      ──┤  (5-fold CV)
      ├──► CatBoost                     ──┤
      └──► Neural Net MLP               ──┘
                                          │
                                          ▼
                               Ridge мета-леарнер (стекинг)
                                          │
                                          ▼
                               submission_stacking.csv
```

---

## Результаты (5-fold CV)

| Модель      | CV RMSLE |
|-------------|----------|
| **Stacking**    | **0.108** |
| ElasticNet  | 0.111    |
| Lasso       | 0.112    |
| CatBoost    | 0.113    |
| Ridge       | 0.113    |
| LightGBM    | 0.122    |
| XGBoost     | 0.123    |

---

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение всех моделей + генерация submission
python train.py

# Инференс из уже обученных моделей (без переобучения)
python predict.py
```

После `train.py` в папке `models/` появляются:
- `*_models.pkl` — обученные fold-модели каждого алгоритма
- `stacking_meta.pkl` — Ridge мета-леарнер
- `oof_predictions.csv` — OOF предсказания + residuals
- `cv_scores.json` — CV RMSLE всех моделей
- `feature_importances.csv` — важность признаков (усреднённая по LightGBM / XGBoost / CatBoost)

---

## Зависимости

Python 3.10+, основные библиотеки:

```
scikit-learn, lightgbm, xgboost, catboost, torch, omegaconf
```
