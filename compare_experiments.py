import argparse
import json
from pathlib import Path
import mlflow
from src.config import load_config

def compare(config):
    mlflow.set_tracking_uri(config["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[config["experiment"]],
        filter_string="attributes.status = 'FINISHED' AND tags.evaluation_split = 'validation'",
        order_by=["metrics.roc_auc DESC"])
    if runs.empty:
        raise RuntimeError("No completed validation experiments")
    cols = ["run_id", "metrics.roc_auc", "metrics.accuracy", "metrics.f1",
            "params.model.n_estimators", "params.model.max_depth", "tags.data_version"]
    Path("reports").mkdir(exist_ok=True)
    runs[cols].to_csv("reports/experiments.csv", index=False)
    best = runs.iloc[0]
    winner = dict(config)
    winner["model"] = dict(config["model"], n_estimators=int(best["params.model.n_estimators"]),
                           max_depth=int(best["params.model.max_depth"]),
                           min_samples_leaf=int(best["params.model.min_samples_leaf"]))
    import yaml
    Path("configs/best.yaml").write_text(yaml.safe_dump(winner, sort_keys=False))
    print(runs[cols].to_string(index=False))
    print("Best validation run:", best.run_id)
    return winner

if __name__ == "__main__":
    compare(load_config())
