import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["age", "tenure", "monthly_charge", "support_calls", "usage_hours", "late_payments"]
CATEGORICAL = ["contract", "payment", "region", "plan"]
FEATURES = NUMERIC + CATEGORICAL

def validate_features(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty pandas DataFrame")
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in NUMERIC):
        raise ValueError("Numeric features must have numeric dtypes")
    if np.isinf(frame[NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric values are invalid")
    return frame.loc[:, FEATURES].copy(deep=True)

def build_preprocessor():
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
