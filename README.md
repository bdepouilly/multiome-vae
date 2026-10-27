# multiome-vae

A variational autoencoder that learns a joint latent representation of single cells from paired gene expression (scRNA-seq) and chromatin accessibility (scATAC-seq) data. It is trained on the 10x Genomics PBMC Multiome dataset. The quality of the latent space is measured by how well a k-nearest-neighbour classifier recovers cell types from it, compared against PCA/LSI baselines.

> **Status:** work in progress. The current model uses ATAC only as an encoder input: the ATAC reconstruction term is switched off (`lambda_atac = 0`) until ATAC gets a proper count likelihood (see [Status and next steps](#status-and-next-steps)).

<!-- TODO: one or two sentences on motivation: why a joint RNA+ATAC embedding, what question this project asks. -->

I started this project as a hands-on way to learn about multimodal variational encoders. However, after familiarizing myself with the field of developmental biology, it became a potential way by which one could formalize Waddington's epigenetic landscape metaphor. More precisely, I want to learn a manifold from genetic (RNA) and epigenetic (ATAC) data, metrize it, and fit a stochastic diffusion model on it that would allow for the prediction of cell fate. It is now the basis of a future research project of mine.

## Contents

- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Method](#method)
- [Evaluation](#evaluation)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Status and next steps](#status-and-next-steps)

## Data

The project uses the **10k Human PBMCs, Multiome v1.0, Chromium X** dataset from 10x Genomics (Cell Ranger ARC output). Download the *Filtered feature barcode matrix (HDF5)* from the 10x Genomics datasets page and place it at:

```
data/raw/10k_PBMC_Multiome_nextgem_Chromium_X_filtered_feature_bc_matrix.h5
```

The file contains both modalities for the same cells: gene expression counts and peak counts.

`data/raw/`, `data/processed/`, `runs/` and `*.h5`/`*.h5ad` files are git-ignored; everything in them can be regenerated with the steps below.

## Installation

Developed with Python 3.14. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the direct dependencies to the versions used during development. The plotting and clustering packages (`matplotlib`, `seaborn`, `umap-learn`, `leidenalg`, `igraph`) are only needed for the notebooks.

## Usage

All scripts are run from the repository root and read/write paths relative to it.

### 1. Preprocess the data

```bash
python src/data/data_processing.py
```

- Splits the 10x matrix into RNA and ATAC.
- **RNA:** QC filtering (minimum genes per cell, MAD-based outliers on counts/genes/top-50-gene share, mitochondrial fraction > 10 %), library-size normalisation to 10⁴ and `log1p`. Raw counts are kept for the count likelihood.
- **Cell-type labels:** assigned with CellTypist on the normalised RNA. There are two types of labels, the "coarse" ones (T, B, NK, monocytes, dendritic cells and "Other") and the "fine" ones, which include more detailed subgroups of the former.
- **Gene selection:** the top 5,000 highly variable genes.
- **ATAC:** cells are aligned to the RNA barcodes, peaks open in fewer than 20 cells are dropped, then TF-IDF and a 256-component truncated SVD (LSI).
- Writes `data/processed/multiome_dataset.npz`, which holds log-normalised RNA, raw RNA counts, ATAC LSI, barcodes, genes, and coarse and fine labels. It also writes the processed `.h5ad` files.

### 2. Split cells into train / validation / test

```bash
python src/data/split_data.py
```

This makes a stratified 80 / 10 / 10 split on the coarse labels (seed 42) and saves it to `data/processed/multiome_split.npz`. All later steps use this split.

### 3. (Experimental) Split-aware ATAC features

```bash
python src/data/atac_features.py
```

- Converts Tn5 insertion counts to fragment counts (`ceil(x / 2)`) following the analysis from Martens, L.D., Fischer, D.S., Yépez, V.A. et al. Modeling fragment counts improves single-cell ATAC-seq analysis. Nat Methods 21, 28–31 (2024). https://doi.org/10.1038/s41592-023-02112-6.
- Keeps peaks open in at least 2 % of *training* cells. This is done after the split to avoid features from the val and test splits to be used for learning.
- Fits TF-IDF and LSI on training cells only.
- Writes `atac_fragments.npz` and `atac_features.npz`. These are not yet consumed by the training script.

### 4. Baselines

```bash
python src/baselines.py
```

This scores three embeddings with the kNN probe and writes `results/baselines.csv`:

- **RNA PCA:** 50 components, fitted on training cells.
- **ATAC LSI:** components 2–51. The first component, which tracks sequencing depth, is dropped.
- **RNA and ATAC:** the two blocks above, concatenated after scaling each to unit standard deviation.

### 5. Train the multimodal VAE

```bash
python src/train_mvae.py --rna-likelihood nb --beta-max 1.0 --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--rna-likelihood` | `nb` | `nb`: negative binomial on raw counts; `gaussian`: MSE on log-normalised expression |
| `--beta-max` | `1.0` | Final KL weight after annealing |
| `--epochs` | `50` | Maximum number of epochs |
| `--patience` | `20` | Early stopping patience (counted only after annealing ends) |
| `--seed` | `42` | Random seed |

Other hyperparameters are set at the top of the script:

- latent dimension 32
- hidden layers `[128, 64]` per modality
- AdamW with learning rate 1e-3, batch size 128
- linear β annealing over 30 epochs (this was included to mitigate posterior collapse)
- `lambda_rna = 1`, `lambda_atac = 0`

Each run creates a directory `runs/<run_name>/` containing:

- `config.json`: the full configuration
- `model_best.pt` / `model_last.pt`: the best (by validation loss) and last checkpoints
- `latent_split.npz`: latent means of the best model for the train / val / test cells, with their labels
- TensorBoard event files: losses, β schedule, latent statistics and per-dimension latent spread

The run name encodes the hyperparameters:

```
mvae_<likelihood>_ld<latent_dim>_bmax<beta_max>_lrna<lambda_rna>_latac<lambda_atac>_lr<lr>_seed<seed>_<YYYYmmdd_HHMMSS>
```

Monitor training with:

```bash
tensorboard --logdir runs
```

`src/train_rna.py` trains an earlier RNA-only VAE (`src/models/rna_vae.py`) and is kept for reference.

### 6. Evaluate trained runs

```bash
python src/eval.py <run_name> [<run_name> ...]
```

This scores each run's latent space with the same kNN probe as the baselines and writes `results/VAE_<run_name>.csv`. It refuses to score a run whose saved labels no longer match the current split, so stale runs cannot be compared silently.

### Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_explorations.ipynb` | Data exploration and QC |
| `notebooks/02_baselines.ipynb` | Baseline embeddings and clustering (UMAP / Leiden) |
| `notebooks/03_comparisons.ipynb` | Comparison of VAE runs against baselines |

<!-- TODO: check these descriptions match the notebooks. -->

## Method

### Architecture

- Each modality has its own MLP encoder (`Linear → LeakyReLU`, hidden sizes `[128, 64]`).
- The two encodings are concatenated, and linear heads produce the mean and log-variance of a shared Gaussian posterior *q(z | x_rna, x_atac)*.
- Each modality is decoded from *z* by its own MLP.

```
x_rna  (5000 HVGs, log-norm) ─► RNA encoder ─┐
                                              ├─► concat ─► μ, log σ² ─► z (32-d) ─┬─► RNA decoder  ─► NB(μ = ℓ·ρ(z), θ)
x_atac (50 LSI comps)        ─► ATAC encoder ─┘                                    └─► ATAC decoder ─► x̂_atac
```

### RNA likelihood

With `--rna-likelihood nb`:

- The decoder outputs gene proportions ρ(z) through a softmax over genes.
- These are scaled by the cell's observed library size ℓ to give the negative-binomial mean μ = ℓ·ρ(z).
- The inverse dispersion θ is a learned per-gene parameter, shared across cells.
- The reconstruction term is the negative NB log-likelihood of the **raw counts**, summed over genes.

The `gaussian` option instead uses MSE on log-normalised expression.

### ATAC input

- ATAC enters the model as LSI components 2–51; component 1 correlates with sequencing depth.
- It is rescaled so that its mean per-feature variance matches that of RNA, using training-cell statistics. This puts both modalities on the same scale when MSE is used for both.

### Loss

```
L = λ_rna · recon_rna + λ_atac · recon_atac + β · KL(q(z|x) ‖ N(0, I))
```

- β is annealed linearly from 1e-5 to `beta_max` over the first 30 epochs.
- Validation loss is always computed at `beta_max`, so it is comparable across epochs.

## Evaluation

The latent space is scored with a **kNN probe**:

- A 15-nearest-neighbour classifier is fitted on training-cell embeddings and evaluated on validation and test cells.
- It reports accuracy, balanced accuracy and macro F1.
- **Coarse labels:** "Other" is excluded.
- **Fine labels:** only cell types with at least 50 training cells are scored.

Model choices are made on the validation split; the test split is reserved for final reporting.

## Results

Validation split, kNN probe. VAE rows are mean ± std over three seeds (42, 1011101, 19102026).

| Embedding | Coarse bal. acc. | Coarse macro F1 | Fine bal. acc. | Fine macro F1 |
|---|---|---|---|---|
| RNA PCA (50) | 0.916 | 0.940 | 0.677 | 0.708 |
| ATAC LSI (2–51) | 0.820 | 0.840 | 0.505 | 0.515 |
| RNA PCA + ATAC LSI | 0.916 | 0.940 | 0.678 | 0.705 |
| MVAE, Gaussian RNA, β_max = 0.001 | 0.897 ± 0.024 | 0.928 ± 0.018 | 0.577 ± 0.018 | 0.588 ± 0.015 |
| MVAE, NB RNA, β_max = 1.0 | 0.897 ± 0.007 | 0.933 ± 0.005 | 0.657 ± 0.018 | 0.689 ± 0.015 |

- Switching the RNA reconstruction from MSE to a negative-binomial likelihood on raw counts improves fine-grained cell-type separation substantially (+8 points of balanced accuracy).
- The VAE does not yet beat the RNA PCA baseline. This is somewhat expected because on RNA-derived labels (CellTypist), RNA PCA is close to optimal, by construction. The MVAE with NB likelihood closes most of the gap.

Full per-run tables are in `results/`.

<!-- TODO: add a UMAP of the latent space coloured by cell type, and the loss curves (total_loss.png / component_losses.png). -->

## Repository structure

```
src/
├── data/
│   ├── data_processing.py   # QC, normalisation, labelling, HVGs, ATAC TF-IDF + LSI
│   ├── split_data.py        # stratified train/val/test split
│   ├── atac_features.py     # split-aware ATAC fragments and LSI (experimental)
│   ├── dataset.py           # RNADataset, PairedMultiomeDataset
│   └── dataloader.py        # loaders and encode_loader (latent means for a loader)
├── models/
│   ├── base_vae.py          # abstract VAE interface
│   ├── rna_vae.py           # RNA-only VAE
│   └── rna_atac_mvae.py     # multimodal RNA + ATAC VAE
├── modules/
│   ├── encoder.py           # MLP encoder
│   ├── decoder.py           # MLP decoder and negative-binomial decoder
│   └── likelihoods.py       # NB log-likelihood
├── train_rna.py             # train the RNA-only VAE
├── train_mvae.py            # train the multimodal VAE
├── baselines.py             # PCA / LSI baselines
├── evaluation.py            # kNN probe and helpers
└── eval.py                  # score trained runs
notebooks/                   # exploration, baselines, comparisons
results/                     # kNN probe scores (CSV)
runs/                        # checkpoints, latents, TensorBoard logs (git-ignored)
data/                        # raw and processed data (git-ignored)
```

## Status and next steps

- **ATAC reconstruction is off** (`lambda_atac = 0`). ATAC currently shapes the latent space only through the encoder. The next step is a count likelihood for ATAC on fragment counts, which `atac_features.py` prepares.
- **The ATAC LSI used for training is fitted on all cells** (`data_processing.py`). The split-aware version in `atac_features.py`, fitted on training cells only, is not wired into training yet.
- Hyperparameters other than the likelihood, β_max and the seed are hard-coded in `train_mvae.py`.

<!-- TODO: references (e.g. scVI / totalVI / MultiVI for the NB decoder design, Martens et al. 2023 for insertion → fragment counts), license, contact. -->
