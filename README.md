# multiome-vae

A variational autoencoder that learns a joint latent representation of single cells from paired gene expression (scRNA-seq) and chromatin accessibility (scATAC-seq) data, trained on the 10x Genomics PBMC Multiome dataset. Both modalities are modelled as **counts**: a negative binomial likelihood for RNA and a Poisson likelihood for ATAC fragment counts. The latent space is evaluated by how well a k-nearest-neighbour classifier recovers cell types from it, against PCA/LSI baselines.

I started this project as a hands-on way to learn about multimodal variational autoencoders. After familiarising myself with developmental biology, it became a first step towards formalising Waddington's epigenetic landscape: learning a manifold from transcriptional (RNA) and epigenetic (ATAC) data, equipping it with a metric, and fitting a stochastic dynamical model on it to predict cell fate. It is the basis of a future research project of mine.

## Contents

- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Method](#method)
- [Evaluation](#evaluation)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Limitations and next steps](#limitations-and-next-steps)
- [References](#references)
- [License](#license)
- [Contact](#contact)

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

Scripts can be run from the repository root; all paths are resolved relative to it.

### 1. Preprocess the data

```bash
python src/data/data_processing.py
```

- Splits the 10x matrix into RNA and ATAC.
- **RNA:** QC filtering (minimum genes per cell, MAD-based outliers on counts/genes/top-50-gene share, mitochondrial fraction > 10 %), library-size normalisation to 10⁴ and `log1p`. Raw counts are kept for the count likelihood.
- **Cell-type labels:** assigned with CellTypist on the normalised RNA, at two resolutions: "coarse" (T, B, NK, monocytes, dendritic cells and "Other") and "fine" (subtypes of the former).
- **Gene selection:** the top 5,000 highly variable genes.
- **ATAC:** cells are aligned to the RNA barcodes and the raw peak counts are stored. (This script also computes an LSI on all cells, which is no longer used: see step 3.)
- Writes `data/processed/multiome_dataset.npz` (log-normalised RNA, raw RNA counts, barcodes, genes, coarse and fine labels) and the processed `.h5ad` files.

### 2. Split cells into train / validation / test

```bash
python src/data/split_data.py
```

This makes a stratified 80 / 10 / 10 split on the coarse labels (seed 42) and saves it to `data/processed/multiome_split.npz`. All later steps use this split.

### 3. Split-aware ATAC features

```bash
python src/data/atac_features.py
```

Everything that is *learned* from the ATAC data is fitted on training cells only, so that validation and test cells never influence the features:

- Converts Tn5 insertion counts to fragment counts with `ceil(x / 2)`: Cell Ranger ARC counts both ends of a fragment, so a fragment with both cut sites in a peak is counted twice (Martens et al., 2024).
- Keeps peaks with at least one fragment in ≥ 2 % of *training* cells (66,118 peaks).
- Fits the TF-IDF weights and a 51-component truncated SVD (LSI) on training cells, then projects all cells.
- Writes `atac_fragments.npz` (sparse cells × peaks fragment counts, the target of the ATAC likelihood) and `atac_features.npz` (LSI, the ATAC encoder input).

### 4. Baselines

```bash
python src/baselines.py
```

This scores three embeddings with the kNN probe and writes `results/baselines.csv`:

- **RNA PCA:** 50 components, fitted on training cells.
- **ATAC LSI:** components 2–51 of the split-aware LSI from step 3.
- **RNA and ATAC:** the two blocks above, concatenated after scaling each to unit standard deviation.

### 5. Train the multimodal VAE

```bash
python src/train_mvae.py --rna-likelihood nb --atac-likelihood poisson --epochs 200 --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--rna-likelihood` | `nb` | `nb`: negative binomial on raw counts; `gaussian`: MSE on log-normalised expression |
| `--atac-likelihood` | `poisson` | `poisson`: Poisson on fragment counts per peak; `gaussian`: MSE on the LSI components |
| `--lambda-rna` | `1.0` | Weight of the RNA reconstruction term |
| `--lambda-atac` | 1 for `poisson`, 0 for `gaussian` | Weight of the ATAC reconstruction term (0 = ATAC is used only as encoder input) |
| `--beta-max` | `1.0` | Final KL weight after annealing |
| `--epochs` | `50` | Maximum number of epochs |
| `--patience` | `20` | Early stopping patience (counted only after annealing ends) |
| `--seed` | `42` | Random seed |

Other hyperparameters are set at the top of the script:

- latent dimension 32
- hidden layers `[128, 64]` per modality
- AdamW with learning rate 1e-3, batch size 128
- linear β annealing over 30 epochs, to mitigate posterior collapse

Each run creates a directory `runs/<run_name>/` containing:

- `config.json`: the full configuration
- `model_best.pt` / `model_last.pt`: the best (by validation loss) and last checkpoints
- `latent_split.npz`: latent means of the best model for the train / val / test cells, with their labels
- TensorBoard event files: losses per modality, β schedule, latent statistics, per-dimension latent spread and hyperparameters

The run name encodes the hyperparameters:

```
mvae_<rna_likelihood>_<atac_likelihood>_ld<latent_dim>_bmax<beta_max>_lrna<lambda_rna>_latac<lambda_atac>_lr<lr>_seed<seed>_<YYYYmmdd_HHMMSS>
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
| `notebooks/03_comparisons.ipynb` | Early experiments (MSE likelihoods): posterior collapse, how the two modalities mix in the latent space, and a sweep over the reconstruction weights λ |

## Method

### Architecture

- Each modality has its own MLP encoder (`Linear → LeakyReLU`, hidden sizes `[128, 64]`).
- The two encodings are concatenated, and linear heads produce the mean and log-variance of a shared Gaussian posterior *q(z | x_rna, x_atac)*.
- Each modality is decoded from *z* by its own MLP.

```
x_rna  (5000 HVGs, log-norm) ─► RNA encoder ─┐                                     ┌─► RNA decoder  ─► NB(ℓ_rna · ρ_rna(z), θ)       on 5,000 gene counts
                                              ├─► concat ─► μ, log σ² ─► z (32-d) ─┤
x_atac (50 LSI comps)        ─► ATAC encoder ─┘                                     └─► ATAC decoder ─► Poisson(ℓ_atac · ρ_atac(z))  on 66,118 peak fragment counts
```

The encoders see normalised, low-dimensional inputs (log-normalised HVGs, LSI); the decoders model the raw counts. This follows scVI (Lopez et al., 2018) and its multimodal extensions (Ashuach et al., 2023).

### Count decoders

Both count decoders share one design (`CountDecoder` in `src/modules/decoder.py`):

- The MLP outputs logits over features, turned into proportions ρ(z) by a softmax, so each cell's predicted profile sums to one.
- The proportions are scaled by the cell's **observed** library size ℓ (total counts for RNA, total fragments in the kept peaks for ATAC): μ = ℓ·ρ(z). Sequencing depth is therefore a covariate, not something the latent space has to encode.
- **RNA:** negative binomial with mean μ and a learned per-gene inverse dispersion θ, shared across cells.
- **ATAC:** Poisson with mean μ on fragment counts. Martens et al. (2024) show that fragment counts are well described by a Poisson model, and that modelling counts outperforms binarising peaks. The Poisson is the θ → ∞ limit of the NB, which is verified numerically.

The log-likelihoods are implemented in `src/modules/likelihoods.py` in log space (`logaddexp`, `lgamma`) for numerical stability. `src/check_likelihoods.py` checks them against `torch.distributions`, checks that the NB converges to the Poisson as θ grows (the gap shrinks as 1/θ), and checks that the decoder means sum to the library size.

### Loss

```
L = λ_rna · recon_rna + λ_atac · recon_atac + β · KL(q(z|x) ‖ N(0, I))
```

- With the count likelihoods, each reconstruction term is the negative log-likelihood **summed over features** and averaged over cells. With λ_rna = λ_atac = β = 1, L is the negative ELBO in nats per cell.
- β is annealed linearly from 1e-5 to `beta_max` over the first 30 epochs.
- Validation loss is always computed at `beta_max`, so it is comparable across epochs; it is used for checkpoint selection and early stopping.

Because it sums over 66k peaks, the ATAC term is about 28× larger than the RNA term after training (≈ 24,000 vs ≈ 870 nats per cell on validation). The ELBO is kept unweighted for now; how much each modality should drive the latent space is an open question (see [Limitations](#limitations-and-next-steps)).

### ATAC encoder input

- ATAC enters the encoder as LSI components 2–51. The first component of the uncentred SVD is close to the mean accessibility profile and nearly constant across cells, so it is dropped.
- It is rescaled so that its mean per-feature variance matches that of RNA, using training-cell statistics, so both encoder inputs are on a similar scale.

## Evaluation

The latent space is scored with a **kNN probe**:

- A 15-nearest-neighbour classifier is fitted on training-cell embeddings (the latent means) and evaluated on validation and test cells.
- It reports accuracy, balanced accuracy and macro F1.
- **Coarse labels:** "Other" is excluded.
- **Fine labels:** only cell types with at least 50 training cells are scored.

Model choices are made on the validation split; the test split is reserved for final reporting.

## Results

kNN probe, balanced accuracy / macro F1. VAE rows are mean ± std over three seeds (42, 1011101, 19102026). All VAEs use latent dimension 32, β_max = 1 and early stopping on validation loss, except where stated.

**Test split**

| Embedding | Coarse bal. acc. | Coarse macro F1 | Fine bal. acc. | Fine macro F1 |
|---|---|---|---|---|
| RNA PCA (50) | 0.908 | 0.923 | 0.667 | 0.691 |
| ATAC LSI (2–51) | 0.893 | 0.910 | 0.457 | 0.467 |
| RNA PCA + ATAC LSI | 0.908 | 0.923 | 0.669 | 0.686 |
| MVAE, MSE RNA, ATAC input only † | 0.882 ± 0.000 | 0.914 ± 0.004 | 0.593 ± 0.027 | 0.610 ± 0.024 |
| MVAE, NB RNA, ATAC input only | 0.887 ± 0.017 | 0.919 ± 0.015 | 0.661 ± 0.010 | 0.688 ± 0.009 |
| **MVAE, NB RNA + Poisson ATAC** | **0.913 ± 0.017** | **0.933 ± 0.009** | **0.672 ± 0.017** | **0.690 ± 0.010** |

**Validation split**

| Embedding | Coarse bal. acc. | Coarse macro F1 | Fine bal. acc. | Fine macro F1 |
|---|---|---|---|---|
| RNA PCA (50) | 0.916 | 0.940 | 0.677 | 0.708 |
| ATAC LSI (2–51) | 0.844 | 0.865 | 0.473 | 0.480 |
| RNA PCA + ATAC LSI | 0.916 | 0.940 | 0.679 | 0.706 |
| MVAE, MSE RNA, ATAC input only † | 0.897 ± 0.024 | 0.928 ± 0.018 | 0.577 ± 0.018 | 0.588 ± 0.015 |
| MVAE, NB RNA, ATAC input only | 0.898 ± 0.008 | 0.926 ± 0.007 | 0.676 ± 0.023 | 0.701 ± 0.022 |
| **MVAE, NB RNA + Poisson ATAC** | **0.913 ± 0.002** | **0.929 ± 0.005** | **0.698 ± 0.015** | **0.713 ± 0.015** |

† Earlier run: β_max = 0.001 (MSE terms are averaged, not summed) and an ATAC LSI fitted on all cells. The NB run on that older LSI gave the same scores as the NB row within one standard deviation.

What the ablations show:

- **NB vs MSE for RNA:** compared on the same (older) ATAC input, modelling raw counts with a negative binomial instead of MSE on log-normalised data improves fine-grained balanced accuracy by 6–8 points (test 0.593 → 0.656, validation 0.577 → 0.657).
- **Adding the Poisson ATAC likelihood:** the latent space now has to explain chromatin accessibility as well. Coarse balanced accuracy rises from 0.887 to 0.913 on test (0.898 to 0.913 on validation), and fine balanced accuracy rises on both splits. The RNA reconstruction is essentially unchanged (≈ 865 vs ≈ 869 nats per cell on validation), so ATAC adds information without degrading RNA. The KL term grows from ≈ 9 to ≈ 40 nats: the posterior carries more information about each cell.
- **Against the baselines:** the full model matches or slightly exceeds RNA PCA on every metric. The margins are within about one seed standard deviation, so the honest summary is "on par with PCA, with a small edge in balanced accuracy". Since the labels are derived from RNA by CellTypist, RNA PCA is a strong baseline by construction (see [Limitations](#limitations-and-next-steps)).

Per-run tables are in `results/`.

## Repository structure

```
src/
├── data/
│   ├── data_processing.py   # QC, normalisation, CellTypist labels, HVGs, RNA/ATAC alignment
│   ├── split_data.py        # stratified train/val/test split
│   ├── atac_features.py     # fragment counts, peak selection, TF-IDF + LSI fitted on training cells
│   ├── dataset.py           # PairedMultiomeDataset (dict batches; sparse ATAC counts densified per cell)
│   └── dataloader.py        # loaders and encode_loader (latent means for a loader)
├── models/
│   ├── base_vae.py          # abstract VAE interface
│   ├── rna_vae.py           # RNA-only VAE (earlier model)
│   └── rna_atac_mvae.py     # multimodal RNA + ATAC VAE
├── modules/
│   ├── encoder.py           # MLP encoder
│   ├── decoder.py           # MLP decoder; CountDecoder with NB and Poisson subclasses
│   └── likelihoods.py       # NB and Poisson log-likelihoods
├── train_rna.py             # train the RNA-only VAE
├── train_mvae.py            # train the multimodal VAE
├── baselines.py             # PCA / LSI baselines
├── evaluation.py            # kNN probe and helpers
├── eval.py                  # score trained runs
└── check_likelihoods.py     # numerical checks of the likelihoods and count decoder
notebooks/                   # exploration, baselines, early experiments
results/                     # kNN probe scores (CSV); results/archive/ holds early exploratory runs
runs/                        # checkpoints, latents, TensorBoard logs (git-ignored)
data/                        # raw and processed data (git-ignored)
```

## Limitations and next steps

- **Circular labels.** Cell types are assigned by CellTypist on log-normalised RNA, so the kNN probe favours RNA-derived embeddings, and RNA PCA is close to optimal by construction. An evaluation that does not depend on RNA-derived labels (e.g. held-out modality prediction, or labels from an independent annotation) would measure what ATAC adds more fairly.
- **Modality balance.** The ATAC term dominates the ELBO by roughly 28×. Next steps are to compare per-modality reconstruction against unimodal models and to test weighting schemes.
- **ATAC QC.** No cell-level ATAC QC (e.g. TSS enrichment, minimum fragments) or ENCODE blacklist filtering is applied yet.
- **Feature selection.** Highly variable genes are selected on all cells before the split (the ATAC features are already split-aware).
- **Posterior collapse.** Only a fraction of the 32 latent dimensions carry information at β = 1.
- **Towards dynamics.** The longer-term goal is to use the joint latent space for trajectory and fate inference, e.g. with CellRank or optimal-transport methods such as moscot, on a developmental dataset rather than PBMCs.

## References

- Kingma, D. P. & Welling, M. Auto-encoding variational Bayes. *ICLR* (2014).
- Lopez, R., Regier, J., Cole, M. B., Jordan, M. I. & Yosef, N. Deep generative modeling for single-cell transcriptomics. *Nat Methods* 15, 1053–1058 (2018).
- Ashuach, T. et al. MultiVI: deep generative model for the integration of multimodal data. *Nat Methods* 20, 1222–1231 (2023).
- Martens, L. D., Fischer, D. S., Yépez, V. A., Theis, F. J. & Gagneur, J. Modeling fragment counts improves single-cell ATAC-seq analysis. *Nat Methods* 21, 28–31 (2024).
- Domínguez Conde, C. et al. Cross-tissue immune cell analysis reveals tissue-specific features in humans. *Science* 376, eabl5197 (2022). (CellTypist)

## License

Released under the [MIT License](LICENSE).

## Contact

Baptiste Depouilly, baptiste.depouilly@gmail.com
