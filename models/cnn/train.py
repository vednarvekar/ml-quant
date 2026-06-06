import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import CNNModel


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "5M"
SAVE_PATH = BASE_DIR / "models" / "cnn_5m_best.pth"

BATCH_SIZE           = 512
EPOCHS               = 50      # [CHANGED] Increased from 30. With cosine LR decay the model
                                # needs more epochs to fully converge before LR hits eta_min.
LR                   = 1e-3
PATIENCE             = 10      # [CHANGED] Increased from 5. With cosine schedule and label
                                # smoothing, val F1 improves more slowly and in smaller steps.
                                # Patience=5 was cutting training too early.
WARMUP_EPOCHS        = 3       # [NEW] Linear LR warmup for first 3 epochs.
                                # Starting at full LR=1e-3 causes large gradient updates in
                                # epoch 1 that can push the model into a bad basin immediately.
                                # Warmup ramps from LR/10 → LR over 3 epochs, stabilising early training.
STATS_SAMPLE_WINDOWS = 50_000
LABEL_SMOOTHING      = 0.1     # [NEW] Label smoothing on CrossEntropyLoss.
                                # Hard labels (0/1) make the model overconfident and hurt
                                # generalisation on noisy financial data. Smoothing 0.1 means
                                # the target for the correct class is 0.9 instead of 1.0,
                                # forcing the model to stay uncertain and not overfit to label noise.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MarketDataset(Dataset):
    def __init__(self, x_path, y_path, shape, mean=None, std=None):
        self.X    = np.memmap(x_path, dtype=np.float32, mode="r", shape=tuple(shape))
        self.y    = np.memmap(y_path, dtype=np.int64,   mode="r", shape=(shape[0],))
        self.mean = mean
        self.std  = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.X[idx], copy=True)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(1, 0)  # (window, features) → (features, window) for Conv1d
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def make_loader(split, shape, shuffle, mean=None, std=None):
    dataset = MarketDataset(
        DATA_DIR / f"X_{split}.npy",
        DATA_DIR / f"y_{split}.npy",
        shape, mean, std,
    )
    # [CHANGED] Added num_workers=4 and pin_memory. On CPU this still helps by loading
    # the next batch in background while the current one is being processed.
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb  = xb.to(DEVICE), yb.to(DEVICE)
            logits  = model(xb)
            loss    = loss_fn(logits, yb)
            total_loss += loss.item() * yb.size(0)
            all_preds.append(logits.argmax(1).cpu())
            all_targets.append(yb.cpu())

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return total_loss / len(targets), compute_metrics(targets, preds)


def compute_metrics(targets, preds):
    accuracy = (preds == targets).mean()
    rows     = []

    for class_id, name in enumerate(["HOLD", "BUY", "SELL"]):
        tp = np.sum((preds == class_id) & (targets == class_id))
        fp = np.sum((preds == class_id) & (targets != class_id))
        fn = np.sum((preds != class_id) & (targets == class_id))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append((name, precision, recall, f1))

    macro_f1 = float(np.mean([r[3] for r in rows]))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "classes": rows}


def print_report(split, loss, report):
    print(f"{split} | loss {loss:.4f} | acc {report['accuracy']:.3f} | macro F1 {report['macro_f1']:.3f}")
    for name, precision, recall, f1 in report["classes"]:
        print(f"  {name:4s} precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")


def class_weights(labels):
    counts  = np.bincount(np.asarray(labels), minlength=3)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def feature_stats(x_path, shape):
    X            = np.memmap(x_path, dtype=np.float32, mode="r", shape=tuple(shape))
    sample_size  = min(STATS_SAMPLE_WINDOWS, shape[0])
    rng          = np.random.default_rng(44)
    sample_idx   = rng.choice(shape[0], size=sample_size, replace=False)
    sample       = np.asarray(X[sample_idx]).reshape(-1, shape[2])
    mean         = sample.mean(axis=0).astype(np.float32)
    std          = sample.std(axis=0).astype(np.float32)
    std          = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


# [NEW] Linear warmup scheduler.
# For the first WARMUP_EPOCHS epochs, LR scales linearly from LR/10 up to LR.
# After warmup, cosine annealing takes over and decays LR smoothly to eta_min.
# Warmup prevents the large random-weight gradients in epoch 1 from causing
# an unstable jump that the model never recovers from.
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # ramp 0.1 → 1.0 over warmup_epochs
        # cosine decay from 1.0 → eta_min/LR after warmup
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        eta_min_ratio = 1e-5 / LR
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── SETUP ───────────────────────────────────────────────────────────────────
with open(DATA_DIR / "shapes.json", "r") as f:
    shapes = json.load(f)

feature_mean, feature_std = feature_stats(DATA_DIR / "X_train.npy", shapes["train_shape"])

train_loader = make_loader("train", shapes["train_shape"], shuffle=True,  mean=feature_mean, std=feature_std)
val_loader   = make_loader("val",   shapes["val_shape"],   shuffle=False, mean=feature_mean, std=feature_std)
test_loader  = make_loader("test",  shapes["test_shape"],  shuffle=False, mean=feature_mean, std=feature_std)

# [CHANGED] n_features updated to 18 to match new process_data.py feature count
model        = CNNModel(n_features=shapes["train_shape"][2], n_classes=3).to(DEVICE)
train_labels = np.memmap(DATA_DIR / "y_train.npy", dtype=np.int64, mode="r", shape=(shapes["train_shape"][0],))

# [CHANGED] Added label_smoothing=LABEL_SMOOTHING to CrossEntropyLoss
loss_fn   = nn.CrossEntropyLoss(weight=class_weights(train_labels), label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

print(f"Training on {DEVICE}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
print(f"Features: {shapes['train_shape'][2]} | Warmup: {WARMUP_EPOCHS} epochs | Patience: {PATIENCE}")

best_val_f1            = 0.0
epochs_without_improvement = 0

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss  = 0.0
    total       = 0
    all_preds, all_targets = [], []

    progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}")
    for xb, yb in progress:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * yb.size(0)
        total      += yb.size(0)
        preds       = logits.argmax(1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(yb.detach().cpu())
        batch_acc = (preds == yb).float().mean().item()
        progress.set_postfix(loss=f"{train_loss / total:.4f}", batch_acc=f"{batch_acc:.3f}")

    train_loss    = train_loss / total
    train_report  = compute_metrics(torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy())
    val_loss, val_report = evaluate(model, val_loader, loss_fn)

    if val_report["macro_f1"] > best_val_f1:
        best_val_f1                = val_report["macro_f1"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), SAVE_PATH)
        saved = "saved"
    else:
        epochs_without_improvement += 1
        saved = f"no improvement {epochs_without_improvement}/{PATIENCE}"

    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch:02d}/{EPOCHS} {saved} | lr {current_lr:.2e}")
    print_report("Train", train_loss, train_report)
    print_report("Val  ", val_loss, val_report)
    print()

    scheduler.step()

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping: validation macro F1 did not improve for {PATIENCE} epochs.")
        break

# ─── FINAL TEST EVALUATION ───────────────────────────────────────────────────
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
test_loss, test_report = evaluate(model, test_loader, loss_fn)
print_report("Test ", test_loss, test_report)
