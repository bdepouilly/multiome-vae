"""ATAC features that depend on the train/val/test split. Run after split_data.py.

1. Convert Tn5 insertion counts to fragment counts (likelihood target)
2. Keep peaks accessible in at least `min_peak_frac` of training cells
3. Computes the first 51 components of the LSI
"""
import numpy as np
import anndata as ad
from scipy import sparse
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"

atac_in = PROCESSED_DATA_DIR / "adata_atac_processed.h5ad"
dataset_in = PROCESSED_DATA_DIR / "multiome_dataset.npz"
split_in = PROCESSED_DATA_DIR / "multiome_split.npz"

fragments_out = PROCESSED_DATA_DIR / "atac_fragments.npz"
features_out = PROCESSED_DATA_DIR / "atac_features.npz"

min_peak_frac = 0.02 # sets the portion of training cells in which a peak should be observed to be kept


def insertions_to_fragments(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """Cell Ranger ARC counts Tn5 insertion sites: a fragment with both ends in a peak counts twice.
    ceil(x / 2) recovers the number of fragments (Martens et al., 2023)."""
    X = X.copy()
    X.data = np.ceil(X.data / 2)
    return X

def select_peaks(X: sparse.csr_matrix, train_idx: np.ndarray, min_frac: float) -> np.ndarray:
    """Boolean mask of peaks with at least one fragment in >= min_frac of training cells."""
    frac_open = np.asarray((X[train_idx] > 0).mean(axis=0)).ravel()
    return frac_open >= min_frac

def fit_idf(X: sparse.csr_matrix, train_idx: np.ndarray) -> np.ndarray:
    """IDF weight per peak, computed from training cells only. Returns shape (n_peaks,)."""
    n_train_cells = len(train_idx)
    peak_counts = np.array((X[train_idx] > 0).sum(axis=0)).flatten()
    idf = np.log(1 + n_train_cells / (1 + peak_counts))
    return idf

def tf_idf(X: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    """TF-IDF for all cells, using a pre-computed IDF."""
    X_tf = normalize(X, norm="l1", axis=1)
    X_tfidf = X_tf.multiply(idf).tocsr()
    X_tfidf = np.log1p(X_tfidf)
    return X_tfidf


def main():
    atac = ad.read_h5ad(atac_in)
    data = np.load(dataset_in, allow_pickle=True)
    split = np.load(split_in)
    train_idx = split["train_idx"]

    # The split indexes rows of multiome_dataset.npz, so ATAC rows must be in the same order
    assert np.array_equal(atac.obs_names.to_numpy(), data["barcodes"]), "ATAC and RNA cell order differ"

    X_frag = insertions_to_fragments(atac.layers["counts"].tocsr())

    peak_mask = select_peaks(X_frag, train_idx, min_peak_frac)
    X_frag = X_frag[:, peak_mask].astype(np.float32)
    peaks = atac.var_names[peak_mask].to_numpy()

    print(f"Peaks kept: {peak_mask.sum()} / {peak_mask.size} (open in >= {min_peak_frac:.0%} of training cells)")

    idf = fit_idf(X_frag, train_idx)
    X_tfidf = tf_idf(X_frag, idf)
    svd = TruncatedSVD(n_components=51, random_state=0)
    svd.fit(X_tfidf[train_idx])
    print("Running SVD decomposition on ATAC data.")
    X_lsi = svd.transform(X_tfidf).astype(np.float32)


    sparse.save_npz(fragments_out, X_frag)
    np.savez_compressed(features_out, peaks=peaks, barcodes=data["barcodes"], min_peak_frac=min_peak_frac, X_lsi=X_lsi)


if __name__ == "__main__":
    main()
