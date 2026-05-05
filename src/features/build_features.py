"""
Feature engineering pipeline for the Ames Housing dataset.

Порядок шагов:
  1. _add_domain_features  — новые признаки на основе доменного знания
  2. _encode_ordinals      — порядковое кодирование качественных колонок
  3. _fix_skewed_features  — log1p для скошенных числовых признаков
  4. _encode_nominals      — one-hot для оставшихся категорий
  5. align                 — выравнивание колонок train/test после дамми-кодирования

Важно: шаги 3-4 применяются к объединённому train+test, чтобы получить
одинаковое множество колонок и одинаковые трансформации в обоих датасетах.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew


# ---------------------------------------------------------------------------
# Маппинги для порядкового кодирования
# Чем выше число — тем лучше качество. "None" = отсутствует объект.
# ---------------------------------------------------------------------------
ORDINAL_MAP = {
    "ExterQual":    {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond":    {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual":     {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond":     {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "HeatingQC":    {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual":  {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu":  {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "GarageQual":   {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond":   {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC":       {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Fence":        {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Functional":   {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "LotShape":     {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive":   {"N": 0, "P": 1, "Y": 2},
    "Utilities":    {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    "LandSlope":    {"Sev": 1, "Mod": 2, "Gtl": 3},
    "CentralAir":   {"N": 0, "Y": 1},
    "Street":       {"Grvl": 0, "Pave": 1},
    "Alley":        {"None": 0, "Grvl": 1, "Pave": 2},
    "Electrical":   {"Mix": 1, "FuseP": 2, "FuseF": 3, "FuseA": 4, "SBrkr": 5},
}

# Числовые колонки, которые по смыслу — категории (ID класса, год, месяц)
NOMINAL_INT_COLS = ["MSSubClass", "MoSold", "YrSold"]


# ---------------------------------------------------------------------------
# Шаг 1: Доменные признаки
# ---------------------------------------------------------------------------

def _add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Конструирует новые признаки на основе доменного знания о рынке недвижимости.

    Группы признаков:
      - Возраст дома: HouseAge, RemodAge, IsRemodeled, IsNew
      - Суммарные площади: TotalSF, TotalBathrooms, TotalPorchSF
      - Бинарные флаги: HasPool, HasGarage, HasBsmt, Has2ndFloor, HasFireplace
      - Взаимодействия качество × площадь: QualArea, QualTotSF

    Примечание: YrSold, YearBuilt, YearRemodAdd оставляем как есть — деревья
    сами могут найти пороговые разбиения по году, а вот линейным моделям
    разности помогают сильнее.
    """
    df = df.copy()

    # --- Возраст ---
    df["HouseAge"]    = df["YrSold"].astype(int) - df["YearBuilt"]
    df["RemodAge"]    = df["YrSold"].astype(int) - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["IsNew"]       = (df["YrSold"].astype(int) == df["YearBuilt"]).astype(int)

    # --- Суммарные площади ---
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Полуванные считаем за 0.5 — стандартная практика в задачах по недвижимости
    df["TotalBathrooms"] = (
        df["FullBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["HalfBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )

    # --- Бинарные флаги наличия объектов ---
    df["HasPool"]      = (df["PoolArea"]    > 0).astype(int)
    df["HasGarage"]    = (df["GarageArea"]  > 0).astype(int)
    df["HasBsmt"]      = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFloor"]  = (df["2ndFlrSF"]    > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"]  > 0).astype(int)

    # --- Взаимодействия ---
    # OverallQual × площадь — стабильно входят в топ важности у всех бустингов
    df["QualArea"]  = df["OverallQual"] * df["GrLivArea"]
    df["QualTotSF"] = df["OverallQual"] * df["TotalSF"]

    return df


# ---------------------------------------------------------------------------
# Шаг 2: Порядковое кодирование
# ---------------------------------------------------------------------------

def _encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заменяет строковые категории осмысленными порядковыми числами.

    Почему не Label Encoding / OHE для этих колонок:
      - Порядок имеет смысл (Ex > Gd > TA > Fa > Po)
      - Для деревьев это даёт правильный порядок при сплитах
      - Для линейных моделей — монотонность
      - Число признаков не раздувается
    """
    df = df.copy()
    for col, mapping in ORDINAL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Шаг 3: Трансформация скошенных признаков
# ---------------------------------------------------------------------------

def _fix_skewed_features(
    df: pd.DataFrame, threshold: float, skip: set
) -> pd.DataFrame:
    """
    Применяет log1p к числовым признакам с |skewness| > threshold.

    Параметры:
        threshold : обычно 0.75 — стандартный порог для Ames Housing
        skip      : набор колонок, которые не трогать (порядковые признаки,
                    бинарные флаги и т.п. — там log1p не нужен)

    Примечание: вызывается на объединённом train+test, чтобы решение
    о трансформации принималось на полной выборке.
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
    skewness = df[num_cols].apply(lambda x: scipy_skew(x.dropna()))
    skewed   = skewness[skewness.abs() > threshold].index.tolist()
    for col in skewed:
        if (df[col] >= 0).all():
            df[col] = np.log1p(df[col])
    return df


# ---------------------------------------------------------------------------
# Шаг 4: One-hot encoding номинальных категорий
# ---------------------------------------------------------------------------

def _encode_nominals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot кодирование оставшихся категориальных признаков.

    MSSubClass, MoSold, YrSold переводим в строки перед pd.get_dummies,
    чтобы они получили текстовый префикс и не смешались с числовыми.
    drop_first=False — оставляем все дамми: Ridge/Lasso сами справятся
    с мультиколлинеарностью, а деревьям всё равно.
    """
    df = df.copy()
    for col in NOMINAL_INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    return df


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    skew_threshold: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Запускает полный пайплайн feature engineering.

    Параметры:
        train           : обучающий датасет без целевой переменной
        test            : тестовый датасет
        skew_threshold  : порог асимметрии для log1p-трансформации

    Возвращает:
        (X_train, X_test) — выровненные DataFrame с одинаковыми колонками
    """
    ordinal_cols = set(ORDINAL_MAP.keys())

    for step in (_add_domain_features, _encode_ordinals):
        train = step(train)
        test  = step(test)

    n_train  = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = _fix_skewed_features(combined, skew_threshold, skip=ordinal_cols)
    combined = _encode_nominals(combined)

    X_train = combined.iloc[:n_train].reset_index(drop=True)
    X_test  = combined.iloc[n_train:].reset_index(drop=True)

    # После get_dummies тест может иметь меньше колонок (редкие категории)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test
