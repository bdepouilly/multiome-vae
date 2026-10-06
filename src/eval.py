import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

data_dir = Path("/Users/bdepouilly/CompBio/multiome-vae/runs/mvae_ld8_bmax0.001_lrna0_latac1_lr0.001_20260401_144115/latent_split.npz")

mu_data = np.load(data_dir, allow_pickle=True)

Z_train = mu_data["Z_train"]
y_train = mu_data["y_train"]
Z_val = mu_data["Z_val"]
y_val = mu_data["y_val"]
Z_test = mu_data["Z_test"]
y_test = mu_data["y_test"]

normalize_and_cluster = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))

normalize_and_cluster.fit(Z_train, y_train)
y_pred = normalize_and_cluster.predict(Z_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy score of RNA-ATAC VAE:", accuracy)