import torch
from torch.utils.data import Dataset

class RNADataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index]
    
class PairedMultiomeDataset(Dataset):
    """Returns (x_rna, x_atac), or (x_rna, x_atac, x_rna_counts) when raw counts are given.

    x_rna (log-normalized) is the encoder input; x_rna_counts is the target of a count likelihood.
    """
    def __init__(self, X_rna, X_atac, X_rna_counts=None):
        self.X_rna = torch.tensor(X_rna, dtype=torch.float32)
        self.X_atac = torch.tensor(X_atac, dtype=torch.float32)
        self.X_rna_counts = None if X_rna_counts is None else torch.tensor(X_rna_counts, dtype=torch.float32)

        assert len(self.X_rna) == len(self.X_atac)
        assert self.X_rna_counts is None or self.X_rna_counts.shape == self.X_rna.shape

    def __len__(self):
        return len(self.X_rna)

    def __getitem__(self, idx):
        if self.X_rna_counts is None:
            return self.X_rna[idx], self.X_atac[idx]
        return self.X_rna[idx], self.X_atac[idx], self.X_rna_counts[idx]