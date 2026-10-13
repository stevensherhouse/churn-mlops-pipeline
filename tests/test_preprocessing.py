import numpy as np
import pandas as pd
import pytest
from src.preprocessing import build_preprocessor, validate_features, NUMERIC, CATEGORICAL

@pytest.fixture
def frame():
    return pd.read_csv("data/churn.csv").head(100)

def test_missing_numeric(frame):
    frame.loc[0, NUMERIC] = np.nan
    assert np.isfinite(build_preprocessor().fit_transform(frame)).all()

def test_missing_categorical(frame):
    frame.loc[0, CATEGORICAL] = np.nan
    assert np.isfinite(build_preprocessor().fit_transform(frame)).all()

def test_one_hot_encoding(frame):
    proc = build_preprocessor().fit(frame)
    encoded = proc.transform(frame)[:, len(NUMERIC):]
    assert set(np.unique(encoded)) <= {0, 1}
    assert (encoded.sum(axis=1) == len(CATEGORICAL)).all()

def test_no_mutation(frame):
    original = frame.copy(deep=True)
    selected = validate_features(frame)
    build_preprocessor().fit_transform(selected)
    selected.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(original, frame)

def test_unknown_category(frame):
    proc = build_preprocessor().fit(frame)
    new = frame.copy()
    new["contract"] = "unseen"
    assert proc.transform(new).shape == proc.transform(frame).shape

@pytest.mark.parametrize("bad", [None, pd.DataFrame(), pd.DataFrame({"age": [20]})])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        validate_features(bad)

def test_invalid_numeric_type(frame):
    frame["age"] = "invalid"
    with pytest.raises(ValueError, match="dtypes"):
        validate_features(frame)

def test_infinite_value(frame):
    frame.loc[0, "monthly_charge"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        validate_features(frame)
