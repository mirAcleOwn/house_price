import pandas as pd
from pathlib import Path


def load_raw_data(raw_train_path, raw_test_path):
    """ Read the train & test datasets"""
    train = pd.read_csv(raw_train_path)
    test = pd.read_csv(raw_test_path)
    return train, test

def quick_overview(df, name):
    """ Fast preview for analyzing data """
    print(f"=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Integer columns: {df.select_dtypes(include='number').shape[1]}")
    print(f"Categorial columns: {df.select_dtypes(include='object').shape[1]}")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    print(f"Missing Top-10: {missing.head(10)}")