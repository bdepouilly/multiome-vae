from torch.utils.data import DataLoader, random_split
import torch

from data.dataset import PairedMultiomeDataset

def split_and_load(dataset, shuffle: bool = True, batch_size: int = 128, train_pct: float = 0.8):
    n_total = len(dataset)
    n_train = int(train_pct * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, val_loader

def split_and_load_paired(dataset, shuffle: bool = True, batch_size: int = 128, train_pct: float = 0.8):
    n_total = len(dataset)
    n_train = int(train_pct * n_total)
    n_val = n_total - n_train
    
    g = torch.Generator().manual_seed(42)
    
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=g)
    
    if shuffle:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader

class PairedDataLoader(DataLoader):
    def __init__(
        self,
        X_rna=None,
        X_atac=None,
        dataset=None,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        if dataset is None:
            if X_rna is None or X_atac is None:
                raise ValueError("Provide either `dataset` or both `X_rna` and `X_atac`.")
            dataset = PairedMultiomeDataset(X_rna, X_atac)
        elif X_rna is not None or X_atac is not None:
            raise ValueError("Pass either `dataset` or `X_rna`/`X_atac`, not both.")

        generator = kwargs.pop("generator", None)
        if generator is None:
            generator = torch.Generator().manual_seed(seed)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            **kwargs,
        )
    
