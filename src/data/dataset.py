import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import numpy as np

class RNADataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index]
    
class PairedMultiomeDataset(Dataset):
    """Returns a dict per cell with the keys of the inputs that were given:
        "x_rna":         log-normalized RNA, encoder input
        "x_atac":        ATAC features (LSI), encoder input
        "x_rna_counts":  raw RNA counts, target of a count likelihood (optional)
        "x_atac_counts": ATAC fragment counts, target of a count likelihood (optional)

    X_atac_counts is kept as a sparse CSR matrix and densified one row at a time.
    """
    def __init__(self, X_rna, X_atac, X_rna_counts=None, X_atac_counts=None):
        self.X_rna = torch.tensor(X_rna, dtype=torch.float32)
        self.X_atac = torch.tensor(X_atac, dtype=torch.float32)
        self.X_rna_counts = None if X_rna_counts is None else torch.tensor(X_rna_counts, dtype=torch.float32)
        self.X_atac_counts = None if X_atac_counts is None else csr_matrix(X_atac_counts, dtype=np.float32)
        assert len(self.X_rna) == len(self.X_atac)
        assert self.X_rna_counts is None or self.X_rna_counts.shape == self.X_rna.shape
        assert self.X_atac_counts is None or self.X_atac_counts.shape[0] == len(self.X_atac)

    def __len__(self):
        return len(self.X_rna)

    def __getitem__(self, idx):
        item = {"x_rna": self.X_rna[idx], "x_atac": self.X_atac[idx]}
        if self.X_rna_counts is not None:
            item["x_rna_counts"] = self.X_rna_counts[idx]
        if self.X_atac_counts is not None:
            item["x_atac_counts"] = torch.from_numpy(self.X_atac_counts[idx].toarray().ravel())
        return item
