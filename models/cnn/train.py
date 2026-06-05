import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from model import CNNModel

# ─── SIMPLE CONFIGURATION ───────────────────────────────────────────────────
DATA_DIR = Path("./data/processed/5M")
MODEL_SAVE_PATH = Path("./models/cnn_best.pth")
MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)

BATCH_SIZE = 512 
EPOCHS     = 30 
LR         = 3e-4
DEVICE     = torch.device('cpu')

# ─── MINIMAL DATASET LOADER ──────────────────────────────────────────────────
class NativeMemmapDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, shape: list):
        self.X = np.memmap(x_path, dtype=np.float32, mode='r', shape=tuple(shape))
        self.y = np.memmap(y_path, dtype=np.int64, mode='r', shape=(shape[0],))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = np.array(self.X[idx], dtype=np.float32)
        # Permute to layout: (Features, Lookback)
        return torch.from_numpy(x_sample).permute(1, 0), torch.tensor(self.y[idx]).long()

# ─── DATA PREPARATION ────────────────────────────────────────────────────────
with open(DATA_DIR / 'shapes.json', 'r') as sf:
    shapes = json.load(sf)

train_ds = NativeMemmapDataset(DATA_DIR / 'X_train.npy', DATA_DIR / 'y_train.npy', shapes['train_shape'])
val_ds   = NativeMemmapDataset(DATA_DIR / 'X_val.npy',   DATA_DIR / 'y_val.npy',   shapes['val_shape'])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# Simple Automatic Class Balancing Weights
train_labels = np.memmap(DATA_DIR / 'y_train.npy', dtype=np.int64, mode='r', shape=(shapes['train_shape'][0],))
counts = np.bincount(train_labels, minlength=3)
weights = train_labels.size / (3.0 * counts)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# ─── MODEL COMPONENTS ────────────────────────────────────────────────────────
model = CNNModel(n_features=12, n_classes=3).to(DEVICE)

cost_function = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LR, 
    steps_per_epoch=len(train_loader), 
    epochs=EPOCHS
)

# ─── CLEAN TRAINING ENGINE ───────────────────────────────────────────────────
print("Starting training loop...")
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # --- TRAINING STEP ---
    model.train()
    running_train_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False)
    
    for xb, yb in train_bar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = cost_function(logits, yb)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        running_train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_train_loss = running_train_loss / len(train_loader)

    # --- VALIDATION STEP ---
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = cost_function(logits, yb)
            
            running_val_loss += loss.item()
            
            # Simple Quick Accuracy Calculation
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    epoch_val_loss = running_val_loss / len(val_loader)
    val_accuracy = (correct / total) * 100

    print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
    print()

    # Save best model variant based strictly on validation cost
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
