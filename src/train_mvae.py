import numpy as np
from data.dataset import PairedMultiomeDataset
from data.dataloader import split_and_load_paired, PairedDataLoader
import torch
from models.rna_atac_mvae import RNA_ATAC_MVAE
from torch.optim import AdamW
from pathlib import Path


# Load processed RNA and ATAC data

data = np.load("/Users/bdepouilly/CompBio/multiome-vae/data/processed/multiome_dataset.npz", allow_pickle=True)
X_rna = data["X_rna"].astype("float32")
X_atac = data["X_atac"].astype("float32")

dataset = PairedMultiomeDataset(X_rna, X_atac)
train_loader, val_loader = split_and_load_paired(dataset=dataset, shuffle=True, batch_size=128, train_pct=0.8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dims = [5000, 256]
latent_dim = 8
hidden_dims = [[128, 64], [128, 64]]

model = RNA_ATAC_MVAE(input_dims=input_dims, hidden_dims=hidden_dims, latent_dim=latent_dim)

model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)

# Training loop

n_epochs = 50
annealing_epochs = 30
beta_max = 0.001
beta_min = 1e-5

train_loss_list = []
train_recon_rna_list = []
train_recon_atac_list = []
train_kl_list = []
val_loss_list = []
val_recon_rna_list = []
val_recon_atac_list = []
val_kl_list = []

for epoch in range(n_epochs):
    model.train()
    beta = min(beta_max, beta_min + (beta_max - beta_min) * epoch / annealing_epochs) #set beta to increase with the number of epochs
    print(f"beta = {beta:.4f}")
    train_loss = 0.0
    train_recon_rna = 0.0
    train_recon_atac = 0.0
    train_kl = 0.0
    
    # For logging
    mu_epoch = []
    logvar_epoch = []
    
    for x_rna, x_atac in train_loader:
        x_rna = x_rna.to(device)
        x_atac = x_atac.to(device)
        
        batch_size = x_rna.size(0)
        
        optimizer.zero_grad()
        outputs = model(x_rna, x_atac)
        
        mu_epoch.append(outputs[4].detach())
        logvar_epoch.append(outputs[5].detach())
        
        loss, recon_rna, recon_atac, kl_loss = model.loss_function(*outputs, beta=beta)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_size
        train_recon_rna += recon_rna.item() * batch_size
        train_recon_atac += recon_atac.item() * batch_size
        train_kl += kl_loss.item() * batch_size
        
    mu_epoch_cat = torch.cat(mu_epoch, dim=0)
    logvar_epoch_cat = torch.cat(logvar_epoch, dim=0)
        
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_recon_rna /= len(train_loader.dataset)
    train_recon_rna_list.append(train_recon_rna)
    train_recon_atac /= len(train_loader.dataset)
    train_recon_atac_list.append(train_recon_atac)
    train_kl /= len(train_loader.dataset)
    train_kl_list.append(train_kl)
    
    model.eval()
    val_loss = 0.0
    val_recon_rna = 0.0
    val_recon_atac = 0.0
    val_kl = 0.0
    
    with torch.no_grad():
        for x_rna, x_atac in val_loader:
            x_rna = x_rna.to(device)
            x_atac = x_atac.to(device)
            
            batch_size = x_rna.size(0)
            
            outputs = model(x_rna, x_atac)
            
            loss, recon_rna, recon_atac, kl_loss = model.loss_function(*outputs, beta=beta)
            
            val_loss += loss.item() * batch_size
            val_recon_rna += recon_rna.item() * batch_size
            val_recon_atac += recon_atac.item() * batch_size
            val_kl += kl_loss.item() * batch_size
        
    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)
    val_recon_rna /= len(val_loader.dataset)
    val_recon_rna_list.append(val_recon_rna)
    val_recon_atac /= len(val_loader.dataset)
    val_recon_atac_list.append(val_recon_atac)
    val_kl /= len(val_loader.dataset)
    val_kl_list.append(val_kl)
    
    print(
        f"Epoch {epoch+1:03d} | train loss = {train_loss:.4f} | train_kl = {train_kl:.4f} | train_recon_rna = {train_recon_rna:.4f} | train_recon_atac = {train_recon_atac:.4f} |"
        f"val_loss = {val_loss:.4f} | val_kl = {val_kl:.4f} | val_recon_rna = {val_recon_rna:.4f} | val_recon_atac = {val_recon_atac:.4f} | "
        f"mu std = {mu_epoch_cat.std(unbiased=False).item():.4f} | logvar std = {logvar_epoch_cat.std(unbiased=False).item():.4f} | "
        f"sigma = {torch.exp(0.5 * logvar_epoch_cat).mean():.4f}"
    )
    
model.eval()
mu_all = []
path_out = Path("/Users/bdepouilly/CompBio/multiome-vae/out")
npz_out = path_out/"collected_latent_mu_multiome.npz"

full_loader = PairedDataLoader(X_rna, X_atac, batch_size=128, shuffle=False, seed=42)

with torch.no_grad():
    for x in full_loader:
        x_rna = x[0].to(device)
        x_atac = x[1].to(device)
        mu, logvar = model.encode(x_rna, x_atac)
        mu_all.append(mu.detach())

Z = torch.cat(mu_all).numpy()

print(f"Z shape: {Z.shape}", end=" | ")
print("Z mean and standard deviation:", Z.mean(), Z.std(), end=" | ")
print("Z mean standard deviation per dimension:", Z.std(axis=0))
np.savez_compressed(npz_out, Z=Z, cell_type_coarse=data["cell_type_coarse"])

