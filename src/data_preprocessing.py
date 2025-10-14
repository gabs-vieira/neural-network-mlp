import pandas as pd
import numpy as np

def load_and_prepare_data(path: str, target_col='Performance'):
    df = pd.read_csv(path)
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col, 'Player', 'Tm'], errors='ignore').select_dtypes(include=[np.number])
    y = df[target_col].copy()
    labels = sorted(y.dropna().unique())
    mapping = {v: i for i, v in enumerate(labels)}
    y_enc = y.map(mapping).fillna(0).astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0).replace(0, 1.0)
    X_train = ((X_train - mu) / sigma).values
    X_test = ((X_test - mu) / sigma).values
    return X_train, X_test, y_train.values, y_test.values, mapping
