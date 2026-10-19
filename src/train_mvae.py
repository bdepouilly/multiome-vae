import numpy as np
from data.dataset import PairedMultiomeDataset
from data.dataloader import encode_loader
import torch
from models.rna_atac_mvae import RNA_ATAC_MVAE
from torch.optim import AdamW
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Subset, DataLoader
import argparse 

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"

# Sanitizing hyperparameters logging

def _to_hparam_value(value):
    if isinstance(value, (bool, int, float, str, torch.Tensor)):
        return value
    if isinstance(value, Path):
        return str(value)

    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _sanitize_hparams(values):
    return {key: _to_hparam_value(value) for key, value in values.items()}

# Hyperparameters

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibilty")
parser.add_argument("--rna-likelihood", type=str, default="nb", choices=["gaussian", "nb"], help="Likelihood for RNA reconstruction")
parser.add_argument("--beta-max", type=float, default=1.0, help="Maximum beta value for KL annealing")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--patience", type=int, default=20, help="Maximum number of epochs without improvement before stopping")
args = parser.parse_args()

seed = args.seed
rna_likelihood = args.rna_likelihood # "gaussian" (MSE on log-normalized) or "nb" (negative binomial on raw counts)
beta_max = args.beta_max
n_epochs = args.epochs
patience = args.patience

latent_dim = 32
hidden_dims = [[128, 64], [128, 64]]
lr = 1e-3
batch_size = 128
annealing_epochs = 30
beta_min = 1e-5
lambda_rna = 1
lambda_atac = 0 # ATAC reconstruction off until it has its own likelihood (still used as encoder input)

# ATAC input: LSI components [atac_first, atac_last). Component 0 tracks sequencing depth, so it is dropped.
atac_first = 1
atac_last = 51

# Load processed RNA and ATAC data

data = np.load(PROCESSED_DATA_DIR / "multiome_dataset.npz", allow_pickle=True)
X_rna = data["X_rna"].astype("float32")
X_rna_counts = data["X_rna_counts"].astype("float32") if rna_likelihood == "nb" else None
X_atac = data["X_atac"][:, atac_first:atac_last].astype("float32")
split = np.load(PROCESSED_DATA_DIR / "multiome_split.npz")

# Scale ATAC so its mean per-feature variance matches RNA's (statistics from training cells only).
# Each MSE term then starts from the same baseline, so lambda_rna = lambda_atac means equal weight.
train_idx = split["train_idx"]
rna_scale = X_rna[train_idx].var(axis=0).mean() ** 0.5
atac_scale = X_atac[train_idx].var(axis=0).mean() ** 0.5
atac_scale_factor = float(rna_scale / atac_scale)
X_atac = X_atac * atac_scale_factor

input_dims = [X_rna.shape[1], X_atac.shape[1]]

# Logging

run_name = datetime.now().strftime(f"mvae_{rna_likelihood}_ld{latent_dim}_bmax{beta_max}_lrna{lambda_rna}_latac{lambda_atac}_lr{lr}_seed{seed}_%Y%m%d_%H%M%S")
run_dir = RUNS_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=run_dir)

# Logging config

config = {
    "input_dim": input_dims, 
    "latent_dim": latent_dim,
    "hidden_dims": hidden_dims,
    "lr": lr,
    "batch_size": batch_size, 
    "beta_schedule": "linear",
    "beta_min": beta_min,
    "beta_max": beta_max,
    "lambda_rna": lambda_rna,
    "lambda_atac": lambda_atac,
    "rna_likelihood": rna_likelihood,
    "epochs": n_epochs,
    "annealing_epochs": annealing_epochs,
    "seed": seed,
    "atac_lsi_components": [atac_first, atac_last],
    "atac_scaling": "match RNA mean per-feature variance on train cells",
    "atac_scale_factor": atac_scale_factor,
}

with open(run_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)
    
config_text = "\n".join(f"{k}: {v}" for k, v in config.items())
writer.add_text("config", config_text, global_step=0)

# Set seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

dataset = PairedMultiomeDataset(X_rna, X_atac, X_rna_counts)

train_dataset = Subset(dataset, split["train_idx"])
val_dataset = Subset(dataset, split["val_idx"])
test_dataset = Subset(dataset, split["test_idx"])

g = torch.Generator().manual_seed(seed)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNA_ATAC_MVAE(input_dims=input_dims, hidden_dims=hidden_dims, latent_dim=latent_dim, rna_likelihood=rna_likelihood)

model.to(device)

optimizer = AdamW(model.parameters(), lr=lr)

# Training loop

train_loss_list = []
train_recon_rna_list = []
train_recon_atac_list = []
train_kl_list = []
val_loss_list = []
val_recon_rna_list = []
val_recon_atac_list = []
val_kl_list = []

epochs_since_best = 0
best_val_loss = float("inf")

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
    
    for batch in train_loader:
        x_rna, x_atac, *rest = [t.to(device) for t in batch]
        x_rna_counts = rest[0] if rest else None

        batch_size = x_rna.size(0)

        optimizer.zero_grad()
        outputs = model(x_rna, x_atac, x_rna_counts)

        mu_epoch.append(outputs["mu"].detach())
        logvar_epoch.append(outputs["logvar"].detach())

        loss, recon_rna, recon_atac, kl_loss = model.loss_function(outputs, beta=beta, lambda_atac=lambda_atac, lambda_rna=lambda_rna)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_size
        train_recon_rna += recon_rna.item() * batch_size
        train_recon_atac += recon_atac.item() * batch_size
        train_kl += kl_loss.item() * batch_size
        
    mu_epoch_cat = torch.cat(mu_epoch, dim=0)
    mu_std = mu_epoch_cat.std(unbiased=False).item()
    z_std_per_dim = mu_epoch_cat.std(dim=0, unbiased=False).cpu().numpy()
    for i, zstd in enumerate(z_std_per_dim):
        writer.add_scalar(f"latent_std_per_dim/z{i}", zstd, epoch)
        
    logvar_epoch_cat = torch.cat(logvar_epoch, dim=0)
    logvar_std = logvar_epoch_cat.std(unbiased=False).item()
    sigma_mean = torch.exp(0.5 * logvar_epoch_cat).mean().item()
        
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
        for batch in val_loader:
            x_rna, x_atac, *rest = [t.to(device) for t in batch]
            x_rna_counts = rest[0] if rest else None

            batch_size = x_rna.size(0)

            outputs = model(x_rna, x_atac, x_rna_counts)

            # Validate at beta_max so val losses are comparable across epochs during the warm-up
            loss, recon_rna, recon_atac, kl_loss = model.loss_function(outputs, beta=beta_max, lambda_atac=lambda_atac, lambda_rna=lambda_rna)
            
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
    
    torch.save(model.state_dict(), run_dir / "model_last.pt") 
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), run_dir / "model_best.pt")
        epochs_since_best = 0
    elif epoch >= annealing_epochs:
        # Count epochs after annealing epochs where the model hasn't improved
        epochs_since_best += 1
    
    # Log metrics
    writer.add_scalar("loss/train_total", train_loss, epoch)
    writer.add_scalar("loss/val_total", val_loss, epoch)
    writer.add_scalar("loss/train_recon_rna", train_recon_rna, epoch)
    writer.add_scalar("loss/val_recon_rna", val_recon_rna, epoch)
    writer.add_scalar("loss/train_recon_atac", train_recon_atac, epoch)
    writer.add_scalar("loss/val_recon_atac", val_recon_atac, epoch)
    writer.add_scalar("loss/train_kl", train_kl, epoch)
    writer.add_scalar("loss/val_kl", val_kl, epoch)
    writer.add_scalar("schedule/beta", beta, epoch)
    writer.add_scalar("latent/mu_std", mu_std, epoch)
    writer.add_scalar("latent/logvar_std", logvar_std, epoch)
    writer.add_scalar("latent/sigma_mean", sigma_mean, epoch)
    writer.add_scalar("schedule/stopped_epoch", epoch + 1)
    
    print(
        f"Epoch {epoch+1:03d} | train loss = {train_loss:.4f} | train_kl = {train_kl:.4f} | train_recon_rna = {train_recon_rna:.4f} | train_recon_atac = {train_recon_atac:.4f} |"
        f"val_loss = {val_loss:.4f} | val_kl = {val_kl:.4f} | val_recon_rna = {val_recon_rna:.4f} | val_recon_atac = {val_recon_atac:.4f} | "
        f"mu std = {mu_std:.4f} | logvar std = {logvar_std:.4f} | "
        f"sigma = {sigma_mean:.4f}"
    )

    if epochs_since_best >= patience:
        print(f"Early stopping at epoch {epoch + 1}: no val improvement for {patience} epochs. Best val loss: {best_val_loss:.4f}.")
        break

best_model_path = run_dir / "model_best.pt"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

print(f"Loaded best model from {best_model_path} (best val_loss: {best_val_loss:.4f})")

train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

labels = data["cell_type_coarse"]

Z_train = encode_loader(model, train_eval_loader, device)
Z_val = encode_loader(model, val_eval_loader, device)
Z_test = encode_loader(model, test_eval_loader, device)

print(f"Z shape: {Z_train.shape}", end=" | ")
print("Z mean and standard deviation:", Z_train.mean(), Z_train.std(), end=" | ")
Z_best_std_per_dim = Z_train.std(axis=0)
n_active_ld = (Z_best_std_per_dim > 0.1).sum()
print("Z mean standard deviation per dimension:", Z_best_std_per_dim)
print(f"Number of active latent dimensions: {n_active_ld}.")

np.savez_compressed(
    run_dir / "latent_split.npz",
    run_name=run_name,
    Z_train=Z_train,
    y_train=labels[split["train_idx"]],
    Z_val=Z_val,
    y_val=labels[split["val_idx"]],
    Z_test=Z_test,
    y_test=labels[split["test_idx"]]
)

writer.add_hparams(
    _sanitize_hparams(config),
    {
        "hparam/final_val_loss": val_loss,
        "hparam/final_val_kl": val_kl,
        "hparam/best_val_loss": best_val_loss,
    }
)

writer.close()
