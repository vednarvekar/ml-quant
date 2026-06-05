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

BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
PATIENCE = 5
STATS_SAMPLE_WINDOWS = 50_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MarketDataset(Dataset):
    def __init__(self, x_path, y_path, shape, mean=None, std=None):
        self.X = np.memmap(x_path, dtype=np.float32, mode="r", shape=tuple(shape))
        self.y = np.memmap(y_path, dtype=np.int64, mode="r", shape=(shape[0],))
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.X[idx], copy=True)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(1, 0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def make_loader(split, shape, shuffle, mean=None, std=None):
    dataset = MarketDataset(
        DATA_DIR / f"X_{split}.npy",
        DATA_DIR / f"y_{split}.npy",
        shape,
        mean,
        std,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            total_loss += loss.item() * yb.size(0)
            all_preds.append(logits.argmax(1).cpu())
            all_targets.append(yb.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return total_loss / len(targets), metrics(targets, preds)


def metrics(targets, preds):
    accuracy = (preds == targets).mean()
    rows = []

    for class_id, name in enumerate(["HOLD", "BUY", "SELL"]):
        tp = np.sum((preds == class_id) & (targets == class_id))
        fp = np.sum((preds == class_id) & (targets != class_id))
        fn = np.sum((preds != class_id) & (targets == class_id))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append((name, precision, recall, f1))

    macro_f1 = float(np.mean([row[3] for row in rows]))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "classes": rows}


def print_report(split, loss, report):
    print(f"{split} | loss {loss:.4f} | acc {report['accuracy']:.3f} | macro F1 {report['macro_f1']:.3f}")
    for name, precision, recall, f1 in report["classes"]:
        print(f"  {name:4s} precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")


def class_weights(labels):
    counts = np.bincount(np.asarray(labels), minlength=3)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def feature_stats(x_path, shape):
    X = np.memmap(x_path, dtype=np.float32, mode="r", shape=tuple(shape))
    sample_size = min(STATS_SAMPLE_WINDOWS, shape[0])
    sample = np.asarray(X[:sample_size]).reshape(-1, shape[2])
    mean = sample.mean(axis=0).astype(np.float32)
    std = sample.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


with open(DATA_DIR / "shapes.json", "r") as f:
    shapes = json.load(f)

feature_mean, feature_std = feature_stats(DATA_DIR / "X_train.npy", shapes["train_shape"])

train_loader = make_loader("train", shapes["train_shape"], shuffle=True, mean=feature_mean, std=feature_std)
val_loader = make_loader("val", shapes["val_shape"], shuffle=False, mean=feature_mean, std=feature_std)
test_loader = make_loader("test", shapes["test_shape"], shuffle=False, mean=feature_mean, std=feature_std)

model = CNNModel(n_features=12, n_classes=3).to(DEVICE)
train_labels = np.memmap(DATA_DIR / "y_train.npy", dtype=np.int64, mode="r", shape=(shapes["train_shape"][0],))
loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_labels))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print(f"Training on {DEVICE}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

best_val_f1 = 0.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}")
    for xb, yb in progress:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * yb.size(0)
        total += yb.size(0)
        preds = logits.argmax(1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(yb.detach().cpu())
        batch_acc = (preds == yb).float().mean().item()
        progress.set_postfix(loss=f"{train_loss / total:.4f}", batch_acc=f"{batch_acc:.3f}")

    train_loss = train_loss / total
    train_preds = torch.cat(all_preds).numpy()
    train_targets = torch.cat(all_targets).numpy()
    train_report = metrics(train_targets, train_preds)
    val_loss, val_report = evaluate(model, val_loader, loss_fn)

    if val_report["macro_f1"] > best_val_f1:
        best_val_f1 = val_report["macro_f1"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), SAVE_PATH)
        saved = "saved"
    else:
        epochs_without_improvement += 1
        saved = f"no improvement {epochs_without_improvement}/{PATIENCE}"

    print(f"\nEpoch {epoch:02d}/{EPOCHS} {saved}")
    print_report("Train", train_loss, train_report)
    print_report("Val  ", val_loss, val_report)
    print()

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping: validation macro F1 did not improve for {PATIENCE} epochs.")
        break

model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
test_loss, test_report = evaluate(model, test_loader, loss_fn)
print_report("Test ", test_loss, test_report)
