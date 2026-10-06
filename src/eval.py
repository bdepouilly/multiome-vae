"""Score the latent space of one or more VAE runs with the same kNN probe as the baselines.

Usage: python src/eval.py <run_name> [<run_name> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import knn_scores, restrict, frequent_classes, MIN_FINE_TRAIN_CELLS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"


def load_latents(run_name, split, labels_coarse):
    """Return the run's latent means as one array with a row per cell, in dataset order.

    The run stores latents per split, in split-index order. Rows of cells outside
    the split stay NaN, so using them by mistake shows up immediately.
    """
    run = np.load(RUNS_DIR / run_name / "latent_split.npz", allow_pickle=True)

    Z = None
    for part in ["train", "val", "test"]:
        idx = split[f"{part}_idx"]
        Z_part = run[f"Z_{part}"]
        # The run saved its labels at training time. If they differ from the current
        # dataset, the run was trained on an older split or labelling and is stale.
        if not np.array_equal(run[f"y_{part}"], labels_coarse[idx]):
            raise ValueError(f"{run_name}: saved {part} labels do not match the current split. Retrain this run.")
        if Z is None:
            Z = np.full((len(labels_coarse), Z_part.shape[1]), np.nan, dtype=Z_part.dtype)
        Z[idx] = Z_part
    return Z


def evaluate_run(run_name, data, split):
    labels = {"coarse": data["cell_type_coarse"], "fine": data["cell_type"]}
    train_idx = split["train_idx"]
    # Make choices on "val"; look at "test" only once all choices are frozen.
    eval_splits = {"val": split["val_idx"], "test": split["test_idx"]}

    # Same class lists as the baselines, computed from training cells only.
    probe_classes = {
        "coarse": [c for c in np.unique(labels["coarse"][train_idx]) if c != "Other"],
        "fine": frequent_classes(labels["fine"][train_idx], min_count=MIN_FINE_TRAIN_CELLS),
    }

    Z = load_latents(run_name, split, labels["coarse"])

    rows = []
    for label_name, y in labels.items():
        classes = probe_classes[label_name]
        probe_train = restrict(train_idx, y, classes)
        for split_name, eval_idx in eval_splits.items():
            probe_eval = restrict(eval_idx, y, classes)
            scores = knn_scores(Z[probe_train], y[probe_train], Z[probe_eval], y[probe_eval])
            rows.append({"label": label_name, "split": split_name, "method": run_name, **scores})
    return pd.DataFrame(rows)


def main():
    run_names = sys.argv[1:]
    if not run_names:
        sys.exit(__doc__)

    data = np.load(PROCESSED_DATA_DIR / "multiome_dataset.npz", allow_pickle=True)
    split = np.load(PROCESSED_DATA_DIR / "multiome_split.npz")
    RESULTS_DIR.mkdir(exist_ok=True)

    for run_name in run_names:
        table = evaluate_run(run_name, data, split)
        table.to_csv(RESULTS_DIR / f"VAE_{run_name}.csv", index=False)

        print(f"\n=== {run_name}")
        for (label_name, split_name), group in table.groupby(["label", "split"], sort=False):
            print(f"\n{label_name} labels, {split_name}")
            print(group.drop(columns=["label", "split", "method"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
