import numpy as np
import pandas as pd
from evaluation import knn_scores
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"
RUN_NAME = sys.argv[1]
LATENT_PATH = RUNS_DIR / RUN_NAME /  "latent_split.npz"


def main():
    mu_data = np.load(LATENT_PATH, allow_pickle=True)

    Z_train = mu_data["Z_train"]
    y_train = mu_data["y_train"]
    Z_val = mu_data["Z_val"]
    y_val = mu_data["y_val"]
    Z_test = mu_data["Z_test"]
    y_test = mu_data["y_test"]

    pairs = {"train": (Z_train, y_train), "val": (Z_val, y_val), "test": (Z_test, y_test)}

    def drop_other(Z, y):
        keep = y != "Other"
        return Z[keep], y[keep]

    pairs = {name: drop_other(Z, y) for name, (Z, y) in pairs.items()}
    Z_train, y_train = pairs["train"]

    rows = []
    for key in ["val", "test"]:
        Z_eval, y_eval = pairs[key]
        scores = knn_scores(Z_train, y_train, Z_eval, y_eval)
        rows.append({"split": key, "method": RUN_NAME, **scores})

    table = pd.DataFrame(rows)
    for split_name, split_table in table.groupby("split", sort=False):
        print(f"\n{split_name}")
        print(split_table.drop(columns="split").round(3).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / f"VAE_{RUN_NAME}.csv", index=False)
    
if __name__ == "__main__":
    main()