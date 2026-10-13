from copy import deepcopy
from src.config import load_config
from src.train import train
from compare_experiments import compare

if __name__ == "__main__":
    base = load_config()
    for estimators, depth, leaf in [(50, 4, 2), (100, 6, 2), (100, 8, 2), (150, 10, 1), (200, 12, 2)]:
        config = deepcopy(base)
        config["model"].update(n_estimators=estimators, max_depth=depth, min_samples_leaf=leaf)
        train(config)
    winner = compare(base)
    train(winner, final=True)
