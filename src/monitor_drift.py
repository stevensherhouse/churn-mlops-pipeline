import argparse
import json
from pathlib import Path
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.config import load_config
from src.preprocessing import NUMERIC, CATEGORICAL, validate_features
from src.train import split_data

def monitor(config, scenario="drifted"):
    data = pd.read_csv(config["data"])
    reference, _, current, _, _, _ = split_data(data, config)
    current = current.copy(deep=True)
    if scenario == "drifted":
        current["monthly_charge"] += 60
        current["support_calls"] += 8
        current["contract"] = "monthly"
    report = Report(metrics=[DataDriftPreset(drift_share=config["monitoring"]["drift_threshold"])])
    report.run(reference_data=reference, current_data=current,
               column_mapping=ColumnMapping(numerical_features=NUMERIC, categorical_features=CATEGORICAL))
    output = Path(config["monitoring"]["report_dir"])
    output.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output / f"drift_{scenario}.html"))
    result = report.as_dict()
    table = next(m["result"] for m in result["metrics"] if m["metric"] == "DataDriftTable")
    drifted = [name for name, info in table["drift_by_columns"].items() if info["drift_detected"]]
    summary = {"scenario": scenario, "drifted_features": drifted,
               "drift_share": table["share_of_drifted_columns"],
               "threshold": config["monitoring"]["drift_threshold"]}
    (output / f"drift_{scenario}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return int(summary["drift_share"] > summary["threshold"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["stable", "drifted"], default="drifted")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    raise SystemExit(monitor(load_config(args.config), args.scenario))
