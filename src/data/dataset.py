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
    def __init__(self, X_rna, X_atac):
        self.X_rna = torch.tensor(X_rna, dtype=torch.float32)
        self.X_atac = torch.tensor(X_atac, dtype=torch.float32)
        
        assert len(self.X_rna) == len(self.X_atac)
    
    def __len__(self):
        return len(self.X_rna)
    
    def __getitem__(self, idx):
        return self.X_rna[idx], self.X_atac[idx]