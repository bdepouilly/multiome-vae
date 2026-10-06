import numpy as np
from sklearn.model_selection import train_test_split

data = np.load("/Users/bdepouilly/CompBio/multiome-vae/data/processed/multiome_dataset.npz", allow_pickle=True)

labels = data['cell_type_coarse']
idx = np.arange(len(labels))

train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)

val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

np.savez_compressed("/Users/bdepouilly/CompBio/multiome-vae/data/processed/multiome_split.npz",
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx)