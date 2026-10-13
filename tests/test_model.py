import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.train import build_model, split_data
from src.evaluate import evaluate, check_thresholds

@pytest.fixture(scope="module")
def trained():
    config = load_config()
    df = pd.read_csv(config["data"])
    Xtrain, _, Xtest, ytrain, _, ytest = split_data(df, config)
    model = build_model(config).fit(Xtrain.iloc[:900], ytrain.iloc[:900])
    return model, Xtest, ytest

def test_prediction_type_shape(trained):
    model, X, _ = trained
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.issubdtype(pred.dtype, np.integer)
    assert set(pred) <= {0,1}

def test_minimum_performance(trained):
    model, X, y = trained
    check_thresholds(evaluate(model, X, y), {"roc_auc":0.85, "accuracy":0.78, "f1":0.70})

def test_gate_rejects_bad_model():
    with pytest.raises(ValueError, match="Performance gate failed"):
        check_thresholds({"roc_auc":0.5}, {"roc_auc":0.85})
