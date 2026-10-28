"""UMAP of a trained run's latent space, coloured by cell type.

Usage: python src/umap_plot.py <run_name> [<output_name>]
Saves figures/<output_name> (default: umap_<run_name>.png).
"""
import sys
import numpy as np
from pathlib import Path
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation import frequent_classes, MIN_FINE_TRAIN_CELLS

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"


def main():
    run_name = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else f"umap_{run_name}.png"

    data = np.load(PROCESSED_DATA_DIR / "multiome_dataset.npz", allow_pickle=True)
    split = np.load(PROCESSED_DATA_DIR / "multiome_split.npz")
    latent_data = np.load(RUNS_DIR / run_name / "latent_split.npz", allow_pickle=True)
    n_cells = len(data["barcodes"])

    # Reassemble the per-split latent means into one matrix in the original cell order,
    # so that the labels from multiome_dataset.npz line up row by row.
    Z = np.empty((n_cells, latent_data["Z_train"].shape[1]), dtype=np.float32)
    y_fine = data["cell_type"] # fine labels
    y_coarse = data["cell_type_coarse"] # coarse labels
    for part in ["train", "val", "test"]:
        idx = split[f"{part}_idx"]
        Z[idx] = latent_data[f"Z_{part}"] # latent cells
        assert (latent_data[f"y_{part}"] == y_coarse[idx]).all(), f"{part}: latent rows don't match split order"

    # np.empty leaves uninitialised rows, so every cell must be filled by exactly one split
    all_idx = np.concatenate([split[f"{part}_idx"] for part in ("train", "val", "test")])
    assert np.array_equal(np.sort(all_idx), np.arange(n_cells)), "split does not cover every cell exactly once"

    print(f"Z: {Z.shape} {Z.dtype} | coarse types: {len(np.unique(y_coarse))} | fine types: {len(np.unique(y_fine))}")

    umap_seed = 42 # set for reproducibility
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=umap_seed)
    embedding = reducer.fit_transform(Z)
    keep_types = frequent_classes(y_fine[split["train_idx"]], min_count=MIN_FINE_TRAIN_CELLS)
    y_fine_display = np.where(np.isin(y_fine, keep_types), y_fine, "Other")

    # Coarse types in a fixed order with "Other" last; tab10 gives clearly distinct colours for few classes
    coarse_types = [t for t in np.unique(y_coarse) if t != "Other"] + ["Other"]
    palette_coarse = dict(zip(coarse_types, sns.color_palette("tab10", len(coarse_types))))
    palette_coarse["Other"] = "lightgrey"

    # Fine types scored by the kNN probe get a colour, the rest are grey background
    fine_types = list(keep_types) + ["Other"]
    palette_fine = dict(zip(keep_types, sns.color_palette("tab20", len(keep_types))))
    palette_fine["Other"] = "lightgrey"

    panels = [
        ("Coarse cell types", y_coarse, coarse_types, palette_coarse),
        (f"Fine cell types (≥ {MIN_FINE_TRAIN_CELLS} training cells)", y_fine_display, fine_types, palette_fine),
    ]

    # constrained layout reserves room for legends placed outside the axes
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), layout="constrained")
    for ax, (title, labels, order, palette) in zip(axes, panels):
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=labels,
            hue_order=order,
            palette=palette,
            s=3,
            linewidth=0,
            rasterized=True,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, markerscale=4, fontsize=9)

    fig.suptitle("MVAE latent space (NB RNA + Poisson ATAC)")

    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()
