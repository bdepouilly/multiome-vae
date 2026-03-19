import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mu_data = np.load("/Users/bdepouilly/CompBio/multiome-vae/out/collected_latent_mu_multiome.npz", allow_pickle=True)
Z = mu_data["Z"]
labels = mu_data["cell_type_coarse"]

Z_tr, Z_te, y_tr, y_te = train_test_split(Z, labels, test_size=0.2, random_state=42, stratify=labels)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(Z_tr, y_tr)
y_pred = knn.predict(Z_te)
accuracy = accuracy_score(y_te, y_pred)

print("Accuracy score of RNA-ATAC VAE:", accuracy)