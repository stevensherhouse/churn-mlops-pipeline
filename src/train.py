import argparse
import hashlib
import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.config import load_config, flatten
from src.preprocessing import validate_features, build_preprocessor
from src.evaluate import evaluate, check_thresholds

def split_data(df, config):
    X, y = validate_features(df), df[config["target"]]
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=config["test_size"],
                                              random_state=config["seed"], stratify=y)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdev, ydev, test_size=config["validation_size"],
                                                random_state=config["seed"], stratify=ydev)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest

def build_model(config):
    params = config["model"].copy()
    if params.pop("type") != "random_forest":
        raise ValueError("Unsupported model type")
    return Pipeline([("preprocess", build_preprocessor()),
                     ("model", RandomForestClassifier(**params, random_state=config["seed"], n_jobs=1))])

def train(config, final=False):
    df = pd.read_csv(config["data"])
    Xtrain, Xval, Xtest, ytrain, yval, ytest = split_data(df, config)
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment"])
    data_hash = hashlib.md5(Path(config["data"]).read_bytes()).hexdigest()
    with mlflow.start_run(run_name="final-test" if final else "validation-experiment") as run:
        mlflow.log_params(flatten(config))
        mlflow.set_tags({"data_version": data_hash, "evaluation_split": "test" if final else "validation"})
        model = build_model(config)
        model.fit(Xtrain, ytrain)
        Xeval, yeval = (Xtest, ytest) if final else (Xval, yval)
        metrics = evaluate(model, Xeval, yeval)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", signature=infer_signature(Xtrain, model.predict(Xtrain)),
                                 input_example=Xtrain.head(3))
        try:
            check_thresholds(metrics, config["thresholds"])
        except ValueError:
            mlflow.set_tag("performance_gate", "failed")
            raise
        mlflow.set_tag("performance_gate", "passed")
        out = Path(config["artifacts_dir"])
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        (out / "metrics.json").write_text(json.dumps({"run_id": run.info.run_id, **metrics}, indent=2))
        print(json.dumps({"run_id": run.info.run_id, **metrics}, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    train(load_config(args.config), final=args.final)
