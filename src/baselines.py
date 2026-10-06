from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from evaluation import knn_scores, restrict, frequent_classes, MIN_FINE_TRAIN_CELLS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

def rna_pca(X_rna, train_idx, n_components=50, seed=42):
    """Fit PCA on the training cells, then project every cell"""
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_rna[train_idx])
    proj = pca.transform(X_rna)
    return proj
    
def main():
    data = np.load(PROCESSED_DATA_DIR / "multiome_dataset.npz", allow_pickle=True)
    split = np.load(PROCESSED_DATA_DIR / "multiome_split.npz")
    train_idx = split["train_idx"]
    # Make choices on "val"; look at "test" only once all choices are frozen.
    eval_splits = {"val": split["val_idx"], "test": split["test_idx"]}
    
    labels = {"coarse": data["cell_type_coarse"], "fine": data["cell_type"]}

    probe_classes = {
        "coarse": [c for c in np.unique(labels["coarse"][train_idx]) if c != "Other"],
        "fine": frequent_classes(labels["fine"][train_idx], min_count=MIN_FINE_TRAIN_CELLS)
    }

    # Embeddings, each with one row per cell.
    Z_rna = rna_pca(data["X_rna"], train_idx)
    Z_atac_no_depth = data["X_atac"][:, 1:51]    # LSI components 2 to 51, dropping the first
    
    # Scale each block to unit overall std so neither modality dominates the distances
    Z_rna = Z_rna / Z_rna.std()
    Z_atac_no_depth = Z_atac_no_depth / Z_atac_no_depth.std()
    Z_both = np.concatenate((Z_rna, Z_atac_no_depth), axis=1)    # RNA and ATAC side by side

    embeddings = {
        "RNA PCA": Z_rna,
        "ATAC LSI no depth": Z_atac_no_depth,
        "RNA and ATAC": Z_both,
    }
    
    assert all(len(Z) == len(data["cell_type"]) for Z in embeddings.values())

    rows = []
    for label_name, y in labels.items():
        classes = probe_classes[label_name]
        probe_train = restrict(train_idx, y, classes)
        for split_name, eval_idx in eval_splits.items():
            probe_eval = restrict(eval_idx, y, classes)
            for name, Z in embeddings.items():
                scores = knn_scores(Z[probe_train], y[probe_train], Z[probe_eval], y[probe_eval])
                rows.append({"label": label_name, "split": split_name, "method": name, **scores})

    table = pd.DataFrame(rows)
    for (label_name, split_name), split_table in table.groupby(["label", "split"], sort=False):
        print(f"\n{label_name}, {split_name}")
        print(split_table.drop(columns=["label", "split"]).round(3).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "baselines.csv", index=False)


if __name__ == "__main__":
    main()
    