import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


CAT_NONE_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType',
]

NUM_ZERO_COLS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath',
]


def build_imputer(train, target):
    """
    Fit ColumnTransformer на train:
      - CAT_NONE_COLS         constant 'None'
      - NUM_ZERO_COLS         constant 0
      - остальные числовые    median
      - остальные категории   most_frequent
    """
    feature_cols = [c for c in train.columns if c != target]
    df = train[feature_cols]

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    cat_none  = [c for c in CAT_NONE_COLS if c in cat_cols]
    num_zero  = [c for c in NUM_ZERO_COLS  if c in num_cols]
    cat_freq  = [c for c in cat_cols if c not in cat_none]
    num_med   = [c for c in num_cols if c not in num_zero]

    transformer = ColumnTransformer(
        transformers=[
            ('cat_none', SimpleImputer(strategy='constant', fill_value='None'), cat_none),
            ('num_zero', SimpleImputer(strategy='constant', fill_value=0), num_zero),
            ('cat_freq', SimpleImputer(strategy='most_frequent'), cat_freq),
            ('num_med',  SimpleImputer(strategy='median'), num_med),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    transformer.fit(df)
    return transformer


def apply_imputer(transformer, df, target):
    """
    Применяет fitted transformer к df, возвращает DataFrame с теми же именами колонок.
    target — опционально, будет добавлен обратно без изменений.
    """
    feature_cols = [c for c in df.columns if c != target]
    transformed = transformer.transform(df[feature_cols])
    result = pd.DataFrame(transformed, columns=transformer.get_feature_names_out(), index=df.index)

    # ColumnTransformer возвращает numpy object-массив при смешанных типах —
    # принудительно восстанавливаем числовые dtype, иначе get_dummies создаёт
    # по колонке на каждое уникальное значение числовых фич
    def _to_numeric(col):
        try:
            return pd.to_numeric(col)
        except (ValueError, TypeError):
            return col

    result = result.apply(_to_numeric)

    if target and target in df.columns:
        result[target] = df[target].values

    return result