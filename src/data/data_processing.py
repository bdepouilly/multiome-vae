import scanpy as sc
import numpy as np
from typing import Tuple
from anndata import AnnData
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

processed_dir = Path("/Users/bdepouilly/CompBio/multiome-vae/data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

rna_out = processed_dir / "adata_rna_processed.h5ad"
atac_out = processed_dir / "adata_atac_processed.h5ad"

data_path = "/Users/bdepouilly/CompBio/multiome-vae/data/raw/10k_PBMC_Multiome_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"

def load_and_split(path: str) -> Tuple[AnnData, AnnData]:
    print(f"Loading data at {path}")
    adata = sc.read_10x_h5(path, gex_only=False)
    adata.var_names_make_unique()
    
    # Separate gene expression and peaks
    adata_rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    adata_atac = adata[:, adata.var["feature_types"] == "Peaks"].copy()
    
    return adata_rna, adata_atac

def is_outlier(adata, metric: str, nmads: int, genes: bool = False): #if genes is true we filter out genes, otherwise we filter out cells/barcodes
    if genes: m = adata.var[metric]
    else: m = adata.obs[metric]
    mad = np.median(np.abs(m - np.median(m)))
    outlier = (m > np.median(m) + nmads*mad) | (m < np.median(m) - nmads*mad)
    return outlier

def filtering(adata: AnnData, min_genes: int = 100, min_cells: int = 3, nmads_outliers: int = 5, nmads_mt_outliers: int = 5, thresh_mt_pct: int = 10) -> AnnData:
    
    #Filter low-quality cells and under-represented genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    #Compute QC metrics for mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    
    #Identify outliers and mt-outliers based on MADs
    mask = (is_outlier(adata, "log1p_total_counts", nmads_outliers) | is_outlier(adata, "log1p_n_genes_by_counts", nmads_outliers) | is_outlier(adata, "pct_counts_in_top_50_genes", nmads_outliers))
    adata.obs["outlier"] = mask
    mask_mt = is_outlier(adata, "pct_counts_mt", nmads_mt_outliers) | (adata.obs['pct_counts_mt'] > thresh_mt_pct)
    adata.obs['mt_outlier'] = mask_mt
    
    # Remove outliers
    print(f"Number of cells pre-filtering: {adata.n_obs}.")
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

    print(f"Number of cells after filtering: {adata.n_obs}.")
    
    return adata

def align_RNA_ATAC(adata_rna: AnnData, adata_atac: AnnData) -> Tuple[AnnData, AnnData]:
    common_barcodes = adata_rna.obs_names.intersection(adata_atac.obs_names)

    adata_rna = adata_rna[common_barcodes].copy()
    adata_atac = adata_atac[common_barcodes].copy()
    
    return adata_rna, adata_atac

def tf_idf(adata_atac):
    adata_atac.layers["counts"] = adata_atac.X.copy()
    X = adata_atac.X
    X_tf = normalize(X, norm='l1', axis=1)

    n_cells = X_tf.shape[0]
    peak_counts = np.array((X_tf > 0).sum(axis=0)).flatten()
    idf = np.log(1 + n_cells / (1 + peak_counts))
    X_tfidf = X_tf.multiply(idf)
    X_tfidf.data = np.log1p(X_tfidf.data)
    adata_atac.layers["X_tfidf"] = X_tfidf.tocsr()
    
    return adata_atac
    
def SVD(adata_atac, n_components: int = 256):
    X = adata_atac.layers["X_tfidf"]
    svd = TruncatedSVD(n_components)
    
    print("Performing SVD on ATAC data.")
    X_lsi = svd.fit_transform(X)
    print("Done.")
    adata_atac.obsm["X_lsi"] = X_lsi
    adata_atac.uns["lsi"] = {
        "variance_ratio": svd.explained_variance_ratio_,
        "singular_values": svd.singular_values_,
    }

    return adata_atac
    
def main():
    adata_rna, adata_atac = load_and_split(data_path)
    
    adata_rna = filtering(adata_rna)
    
    # Saving count data
    adata_rna.layers['counts'] = adata_rna.X.copy()

    # Normalizing to median total coutns
    sc.pp.normalize_total(adata_rna)
    
    # Log-transform the data
    sc.pp.log1p(adata_rna)
    
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=5000, subset=True)
    
    adata_rna, adata_atac = align_RNA_ATAC(adata_rna, adata_atac)
    
    sc.pp.filter_genes(adata_atac, min_cells=20)
    
    adata_atac = tf_idf(adata_atac)
    adata_atac = SVD(adata_atac)
    
    adata_rna.write(rna_out)
    adata_atac.write(atac_out)

if __name__ == "__main__":
    main()