import argparse
import math
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset  # Switched TensorDataset to base Dataset for disk-mapping

from model import MultiTimeframeCNN

# ── Updated Path Rules ───────────────────────────────────────────────────────
# Go 4 levels up to hit your true root directory (~/ml-quant)
BASE_DIR      = Path(__file__).resolve().parent.parent.parent.parent
MASTER_DIR    = BASE_DIR / "data" / "processed" / "master_splits"
MODEL_OUTPUT  = BASE_DIR / "models" / "cnn" / "cnn_model.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on train/val/test split data.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    return parser.parse_args()


# ── Memory-Mapped Custom Dataset Class ───────────────────────────────────────
class MappedStockDataset(Dataset):
    """
    Custom Dataset that references arrays on your hard drive via memory maps.
    This prevents pulling all 1.08 million records into system RAM simultaneously.
    """
    def __init__(self, prefix):
        # np.load(..., mmap_mode='r') keeps the file open on disk without allocating data arrays to RAM
        self.x3m_map = np.load(MASTER_DIR / f"{prefix}_X_3min.npy", mmap_mode='r')
        self.x5m_map = np.load(MASTER_DIR / f"{prefix}_X_5min.npy", mmap_mode='r')
        self.x1h_map = np.load(MASTER_DIR / f"{prefix}_X_1hr.npy", mmap_mode='r')
        self.y_map   = np.load(MASTER_DIR / f"{prefix}_y_labels.npy", mmap_mode='r')
        self.length  = self.y_map.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extract only the exact index slice requested by the DataLoader batch window
        # .copy() drops read-only memory locks to pass clean PyTorch tensors smoothly
        x3m = torch.from_numpy(self.x3m_map[idx].copy()).float().unsqueeze(0) # Unsqueeze(0) replaces old .unsqueeze(1) for item lookup
        x5m = torch.from_numpy(self.x5m_map[idx].copy()).float().unsqueeze(0)
        x1h = torch.from_numpy(self.x1h_map[idx].copy()).float().unsqueeze(0)
        y   = torch.tensor(self.y_map[idx]).long()
        
        return x3m, x5m, x1h, y


def load_split(prefix):
    """
    prefix will be 'TRAIN', 'VAL', or 'TEST'.
    Make sure your master dataset compilation script saves the combined files 
    using the exact suffixes below matching your 3min, 5min, and 1hr selections.
    """
    # 1. Match 'X_3min' instead of X1 or X3
    # 2. Match 'X_5min' instead of X5
    # 3. Match 'X_1hr' instead of XH
    # 4. Match the labels
    
    # Returns our memory-efficient implementation wrapping the targets
    return MappedStockDataset(prefix)


def get_loaders(batch_size):
    train_dataset = load_split("TRAIN")
    val_dataset = load_split("VAL")
    test_dataset = load_split("TEST")

    # num_workers allows asynchronous asynchronous pre-fetching of slices from your disk storage
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)


def format_time(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(step, total, width=24):
    filled = math.floor((step / total) * width) if total else width
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def compute_metrics(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    accuracy = float((preds == labels).mean()) if len(labels) else 0.0

    recalls = {}
    for class_id in [0, 1, 2]:
        class_mask = labels == class_id
        total_class = int(class_mask.sum())
        if total_class == 0:
            recalls[class_id] = 0.0
        else:
            recalls[class_id] = float((preds[class_mask] == class_id).mean())

    balanced_accuracy = (recalls[0] + recalls[1] + recalls[2]) / 3.0
    return accuracy, balanced_accuracy, recalls


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for b_x3, b_x5, b_xh, b_y in loader:
            b_x3 = b_x3.to(device)
            b_x5 = b_x5.to(device)
            b_xh = b_xh.to(device)
            b_y = b_y.to(device)

            outputs = model(b_x3, b_x5, b_xh)
            loss = criterion(outputs, b_y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(b_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    accuracy, balanced_accuracy, recalls = compute_metrics(all_preds, all_labels)
    return avg_loss, accuracy, balanced_accuracy, recalls


def main():
    args = parse_args()
    device = torch.device(args.device)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, train_size, val_size, test_size = get_loaders(args.batch_size)

    model = MultiTimeframeCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0
    total_start = time.time()

    print(f"Starting training on {device} | train={train_size} val={val_size} test={test_size}")

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_correct = 0
        train_seen = 0
        total_steps = len(train_loader)

        print(f"\nEpoch {epoch}/{args.epochs}")

        for step, (b_x3, b_x5, b_xh, b_y) in enumerate(train_loader, start=1):
            b_x3 = b_x3.to(device)
            b_x5 = b_x5.to(device)
            b_xh = b_xh.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            outputs = model(b_x3, b_x5, b_xh)
            loss = criterion(outputs, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == b_y).sum().item()
            train_seen += b_y.size(0)

            if step == 1 or step % 50 == 0 or step == total_steps:
                elapsed = time.time() - epoch_start
                eta = (elapsed / step) * (total_steps - step)
                avg_train_loss = train_loss_sum / step
                avg_train_acc = train_correct / train_seen if train_seen else 0.0
                
                # Formatted to % by multiplying by 100
                print(
                    f"Epoch {epoch}/{args.epochs} | train {progress_bar(step, total_steps)} "
                    f"{step}/{total_steps} | loss {avg_train_loss * 100:.2f}% | acc {avg_train_acc * 100:.2f}% | "
                    f"elapsed {format_time(elapsed)} | eta {format_time(eta)}"
                )

        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_seen if train_seen else 0.0

        print(f"Epoch {epoch}/{args.epochs} | val   {progress_bar(len(val_loader), len(val_loader))} evaluating")
        val_loss, val_acc, val_bal_acc, val_recalls = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        improved = val_loss < (best_val_loss - args.early_stop_min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            torch.save(model.state_dict(), MODEL_OUTPUT)
            status = "improved"
        else:
            no_improve_count += 1
            status = f"no improve ({no_improve_count}/{args.early_stop_patience})"

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start
        remaining = epoch_time * (args.epochs - epoch)

        # Multiplied metrics by 100 and changed formatting suffix to :.2f% 
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss * 100:.2f}% | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | Val Bal Acc: {val_bal_acc * 100:.2f}% | "
            f"Recall N/B/S: {val_recalls[0] * 100:.2f}%/{val_recalls[1] * 100:.2f}%/{val_recalls[2] * 100:.2f}% | "
            f"Best Val Loss: {best_val_loss * 100:.2f}% @ epoch {best_epoch} | "
            f"status: {status} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"epoch time {format_time(epoch_time)} | total {format_time(total_elapsed)} | "
            f"remaining {format_time(remaining)}"
        )

        if no_improve_count >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best validation loss was {best_val_loss * 100:.2f}% at epoch {best_epoch}.")
            break

    model.load_state_dict(torch.load(MODEL_OUTPUT, map_location=device))
    print(f"\nBest Model Test {progress_bar(len(test_loader), len(test_loader))} evaluating")
    test_loss, test_acc, test_bal_acc, test_recalls = evaluate(model, test_loader, criterion, device)
    
    # Updated final test evaluation print
    print(
        f"Test Loss: {test_loss * 100:.2f}% | Test Acc: {test_acc * 100:.2f}% | Test Bal Acc: {test_bal_acc * 100:.2f}% | "
        f"Recall N/B/S: {test_recalls[0] * 100:.2f}%/{test_recalls[1] * 100:.2f}%/{test_recalls[2] * 100:.2f}%"
    )
    print(f"Best model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()