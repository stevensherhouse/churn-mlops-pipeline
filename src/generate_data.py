"""Deterministic, original synthetic benchmark; no real customer records."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(path="data/churn.csv", seed=2026, rows=3000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 81, rows),
        "tenure": rng.integers(0, 73, rows),
        "monthly_charge": rng.uniform(20, 120, rows).round(2),
        "support_calls": rng.integers(0, 11, rows),
        "usage_hours": rng.uniform(0, 100, rows).round(2),
        "late_payments": rng.integers(0, 6, rows),
        "contract": rng.choice(["monthly", "annual", "two_year"], rows),
        "payment": rng.choice(["card", "bank", "cash"], rows),
        "region": rng.choice(["north", "south", "east", "west"], rows),
        "plan": rng.choice(["basic", "premium"], rows),
    })
    score = (2.8 * (df.contract == "monthly") + 0.5 * df.support_calls
             + 0.65 * df.late_payments - 0.06 * df.tenure
             + 0.012 * df.monthly_charge + rng.normal(0, 0.65, rows))
    df["churn"] = (score > 4).astype(int)
    for col in df.columns.drop("churn"):
        df.loc[rng.random(rows) < 0.03, col] = np.nan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return df

if __name__ == "__main__":
    generate()
