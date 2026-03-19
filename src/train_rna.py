import torch
import numpy as np
from data.dataset import RNADataset
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW
from models.rna_vae import RNA_VAE
import matplotlib.pyplot as plt 
from pathlib import Path

# Load processed RNA matrix
data = np.load("/Users/bdepouilly/CompBio/multiome-vae/data/processed/multiome_dataset.npz", allow_pickle=True)
X_rna = data["X_rna"].astype("float32")

dataset = RNADataset(X_rna)

#Full loader to save data for accuracy measurement
full_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Train/validation split

n_total = len(dataset)
n_train = int(0.8*n_total)
n_val = n_total - n_train

train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model choice

model = RNA_VAE(input_dim=5000, hidden_dims=[128, 64], latent_dim=4).to(device)

#Optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

# Traning loop

n_epochs = 50
annealing_epochs = 40
beta_min = 1e-4
beta_max = 0.01

train_loss_list = []
train_recon_list = []
train_kl_list = []
val_loss_list = []
val_recon_list = []
val_kl_list = []

for epoch in range(n_epochs):
    model.train()
    beta = min(beta_max, beta_min + (beta_max - beta_min) * epoch / annealing_epochs) #set beta to increase with the number of epochs
    print(f"beta = {beta:.4f}")
    train_loss = 0.0
    train_recon = 0.0
    train_kl = 0.0
    
    # For logging
    mu_epoch = []
    logvar_epoch = []
    
    for x in train_loader:
        x = x.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        mu_epoch.append(outputs[2].detach())
        logvar_epoch.append(outputs[3].detach())
        loss, recon_loss, kl_loss = model.loss_function(*outputs, beta=beta)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * x.size(0)
        train_recon += recon_loss.item() * x.size(0)
        train_kl += kl_loss.item() * x.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_recon /= len(train_loader.dataset)
    train_kl /= len(train_loader.dataset)
    
    train_loss_list.append(train_loss)
    train_recon_list.append(train_recon)
    train_kl_list.append(train_kl)
    
    model.eval()
    val_loss = 0.0
    val_recon = 0.0
    val_kl = 0.0
    
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            outputs = model(x)
            loss, recon_loss, kl_loss = model.loss_function(*outputs, beta=beta)
            val_loss += loss.item() * x.size(0)
            val_recon += recon_loss.item() * x.size(0)
            val_kl += kl_loss.item() * x.size(0)
        
    val_loss /= len(val_loader.dataset)
    val_recon /= len(val_loader.dataset)
    val_kl /= len(val_loader.dataset)
    
    val_loss_list.append(val_loss)
    val_recon_list.append(val_recon)
    val_kl_list.append(val_kl)
    
    mu_epoch_cat = torch.cat(mu_epoch, dim=0)
    logvar_epoch_cat = torch.cat(logvar_epoch, dim=0)

    print(
        f"Epoch {epoch+1:03d} | train loss = {train_loss:.4f} | train_kl = {train_kl:.4f} | train_recon = {train_recon:.4f} |"
        f"val_loss = {val_loss:.4f} | val_recon = {val_recon:.4f} | val_kl = {val_kl:.4f} | "
        f"mu std = {mu_epoch_cat.std(unbiased=False).item():.4f} | logvar std = {logvar_epoch_cat.std(unbiased=False).item():.4f} | "
        f"sigma = {torch.exp(0.5 * logvar_epoch_cat).mean():.4f}"
    )
    
epochs = range(1, len(train_loss_list) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss_list, label="Train Total Loss")
plt.plot(epochs, val_loss_list, label="Val Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Total Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("total_loss.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_recon_list, label="Train Recon")
plt.plot(epochs, train_kl_list, label="Train KL")
plt.plot(epochs, val_recon_list, label="Val Recon")
plt.plot(epochs, val_kl_list, label="Val KL")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Reconstruction and KL Loss Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("component_losses.png")
plt.close()

model.eval()
mu_all = []
path_out = Path("/Users/bdepouilly/CompBio/multiome-vae/out")
npz_out = path_out/"collected_latent_mu.npz"

with torch.no_grad():
        for x in full_loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            mu_all.append(mu.cpu())
    
Z = torch.cat(mu_all).numpy()

print(f"Z shape: {Z.shape}", end=" | ")
print("Z mean and standard deviation:", Z.mean(), Z.std(), end=" | ")
print("Z mean standard deviation per dimension:", Z.std(axis=0))
np.savez_compressed(npz_out, Z=Z, cell_type_coarse=data["cell_type_coarse"])
