from pathlib import Path
import yaml

def load_config(path="configs/train.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def flatten(config, prefix=""):
    result = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result
