import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import os
from tqdm import tqdm

from model import CNNModel

# ─── DIRECTORY CONFIGURATION ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / '5M'
MODEL_DIR     = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# ─── TRAINING HYPERPARAMETERS ────────────────────────────────────────────────
BATCH_SIZE  = 512       
EPOCHS      = 30        
LR          = 3e-4      # learning rate for AdamW
PATIENCE    = 7       
SEED        = 44
N_WORKERS   = int(os.getenv('CNN_NUM_WORKERS', '0'))
DEVICE      = torch.device('cpu') # Explicit CPU setup

torch.manual_seed(SEED)


def balanced_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    total = counts.sum()
    weights = np.ones(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = total / (n_classes * counts[present])
    return torch.tensor(weights, dtype=torch.float)


def macro_f1_score(targets, preds, n_classes: int = 3) -> float:
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    f1_values = []

    for class_id in range(n_classes):
        tp = np.sum((preds == class_id) & (targets == class_id))
        fp = np.sum((preds == class_id) & (targets != class_id))
        fn = np.sum((preds != class_id) & (targets == class_id))
        denom = (2 * tp) + fp + fn
        f1_values.append(0.0 if denom == 0 else (2 * tp) / denom)

    print(f"  Per-class F1 — HOLD:{f1_values[0]:.3f}  BUY:{f1_values[1]:.3f}  SELL:{f1_values[2]:.3f}")

    return float(np.mean(f1_values))


# ─── MEMORY-MAPPED REAL-TIME DATA LOADER ─────────────────────────────────────
class NativeMemmapDataset(Dataset):
    """Loads raw binary matrices from disk on demand. Absolute zero RAM profile."""
    def __init__(self, x_path: Path, y_path: Path, shape: list):
        # Open disk handles with specific target shapes
        self.X = np.memmap(x_path, dtype=np.float32, mode='r', shape=tuple(shape))
        self.y = np.memmap(y_path, dtype=np.int64, mode='r', shape=(shape[0],))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Read exactly one lookback window into system RAM
        x_sample = np.array(self.X[idx], dtype=np.float32)
        # Permute from (Lookback, Features) to Channel-First format (Features, Lookback)
        return torch.from_numpy(x_sample).permute(1, 0), torch.tensor(self.y[idx]).long()

# ─── PIPELINE INITIALIZATION ─────────────────────────────────────────────────
# Load shapes metadata to properly unpack binary structures
with open(PROCESSED_DIR / 'shapes.json', 'r') as sf:
    shapes = json.load(sf)

# Initialize Datasets
train_ds = NativeMemmapDataset(PROCESSED_DIR / 'X_train.npy', PROCESSED_DIR / 'y_train.npy', shapes['train_shape'])
val_ds   = NativeMemmapDataset(PROCESSED_DIR / 'X_val.npy',   PROCESSED_DIR / 'y_val.npy',   shapes['val_shape'])
test_ds  = NativeMemmapDataset(PROCESSED_DIR / 'X_test.npy',  PROCESSED_DIR / 'y_test.npy',  shapes['test_shape'])

# Initialize DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

# Dynamically calculate weights from training subset labels to handle heavy class imbalance
train_labels = np.memmap(PROCESSED_DIR / 'y_train.npy', dtype=np.int64, mode='r', shape=(shapes['train_shape'][0],))
class_weights = balanced_class_weights(train_labels, n_classes=3).to(DEVICE)

# Model setup
model     = CNNModel(n_features=12, n_classes=3).to(DEVICE)
cost_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Label smoothing to mitigate noisy financial labels
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# Replace your CosineAnnealingLR with this:
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LR,
    steps_per_epoch=len(train_loader), 
    epochs=EPOCHS,
    pct_start=0.2  # Spend the first 20% of training warming up smoothly
)


# ─── EVALUATION FUNCTION (MACRO F1 CENTRIC) ──────────────────────────────────
def evaluate(model, loader, cost_function):
    """Evaluates metrics globally. Focuses on Macro F1 instead of simple accuracy."""
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluating"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            
            total_loss += cost_function(logits, yb).item() * len(yb)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
            
    avg_loss = total_loss / len(all_targets)
    acc = (np.array(all_preds) == np.array(all_targets)).mean()
    macro_f1 = macro_f1_score(all_targets, all_preds)
    return avg_loss, acc, macro_f1


# ─── ENGINE RUN TIME TRAINING LOOP ───────────────────────────────────────────
print("Starting optimization engine across cross-sectional profiles...\n")
best_val_f1 = -1.0
no_improve = 0  # Counter for Early Stopping tracking

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, all_train_preds, all_train_targets = 0.0, [], []

    for xb, yb in tqdm(train_loader, desc="Training"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        
        logits = model(xb)
        loss   = cost_function(logits, yb)
        loss.backward()
        
        # Hard gradient clipping prevents explosive landscape spikes common in financial series
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        all_train_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_train_targets.extend(yb.cpu().numpy())

    
    # Calculate performance metrics
    t_loss = train_loss / len(train_loader)
    t_acc = (np.array(all_train_preds) == np.array(all_train_targets)).mean()
    t_f1 = macro_f1_score(all_train_targets, all_train_preds)
    
    v_loss, v_acc, v_f1 = evaluate(model, val_loader, cost_function)

    # Performance checkpoint / early stopping logic
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        no_improve = 0  # Reset counter
        torch.save(model.state_dict(), MODEL_DIR / 'cnn_5m_best.pt')
        flag = " ← saved (best val, F1 model state)"
    else:
        no_improve += 1
        flag = f" (no validation improvement {no_improve}/{PATIENCE})"

    print(f"Epoch {epoch:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc * 100:.2f} F1: {t_f1 * 100:.2f} | "
          f"Val Loss: {v_loss:.4f} Acc: {v_acc * 100:.2f} F1: {v_f1 * 100:.2f}{flag}")
    print() 

    # Trigger early stopping step if patience limit is hit
    if no_improve >= PATIENCE:
        print(f"\n[System Alert] Early stopping criteria matched at Epoch {epoch}. Training cut short.")
        break

# Final Verification Evaluation on Unseen Test Dataset Segment
model.load_state_dict(torch.load(MODEL_DIR / 'cnn_5m_best.pt'))
test_loss, test_acc, test_f1 = evaluate(model, test_loader, cost_function)
print(f"\n🚀 Pipeline complete. Unseen Cross-Sectional Test Performance:\n"
      f"   Loss: {test_loss * 100:.2f} | Accuracy: {test_acc * 100:.2f} | Macro-F1 Score: {test_f1 * 100:.2f}")
