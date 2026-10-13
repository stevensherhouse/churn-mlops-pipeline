import pandas as pd
from src.preprocessing import FEATURES

def test_columns_and_size():
    df = pd.read_csv("data/churn.csv")
    assert set(df.columns) == set(FEATURES + ["churn"])
    assert len(df) >= 1000
    assert df[FEATURES].isna().any().any()

def test_target_values():
    df = pd.read_csv("data/churn.csv")
    assert not df.churn.isna().any()
    assert set(df.churn) == {0, 1}

def test_numeric_ranges():
    df = pd.read_csv("data/churn.csv")
    for col, low, high in [("age",18,80),("tenure",0,72),("monthly_charge",20,120),
                            ("support_calls",0,10),("usage_hours",0,100),("late_payments",0,5)]:
        assert df[col].dropna().between(low, high).all(), col
