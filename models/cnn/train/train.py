import argparse
import math
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.cnn.train.model import MultiTimeframeCNN

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DIR = BASE_DIR / "data" / "master_training"
MODEL_OUTPUT = BASE_DIR / "models" / "cnn" / "cnn_model.pth"


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


def load_split(prefix):
    # Each split contains one array per timeframe plus the class labels.
    x1 = torch.from_numpy(np.load(MASTER_DIR / f"{prefix}_X1.npy")).float().unsqueeze(1)
    x5 = torch.from_numpy(np.load(MASTER_DIR / f"{prefix}_X5.npy")).float().unsqueeze(1)
    xh = torch.from_numpy(np.load(MASTER_DIR / f"{prefix}_XH.npy")).float().unsqueeze(1)
    y = torch.from_numpy(np.load(MASTER_DIR / f"{prefix}_y.npy")).long()
    return TensorDataset(x1, x5, xh, y)


def get_loaders(batch_size):
    train_dataset = load_split("TRAIN")
    val_dataset = load_split("VAL")
    test_dataset = load_split("TEST")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    # Balanced accuracy matters here because neutral dominates the dataset.
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
    # Validation and test both use the same evaluation path.
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for b_x1, b_x5, b_xh, b_y in loader:
            b_x1 = b_x1.to(device)
            b_x5 = b_x5.to(device)
            b_xh = b_xh.to(device)
            b_y = b_y.to(device)

            outputs = model(b_x1, b_x5, b_xh)
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
    # Heavier weights on buy/sell reduce the tendency to predict neutral too often.
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

        for step, (b_x1, b_x5, b_xh, b_y) in enumerate(train_loader, start=1):
            b_x1 = b_x1.to(device)
            b_x5 = b_x5.to(device)
            b_xh = b_xh.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            outputs = model(b_x1, b_x5, b_xh)
            loss = criterion(outputs, b_y)
            loss.backward()
            # Clipping keeps unstable batches from exploding the gradients.
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
                print(
                    f"Epoch {epoch}/{args.epochs} | train {progress_bar(step, total_steps)} "
                    f"{step}/{total_steps} | loss {avg_train_loss:.4f} | acc {avg_train_acc:.4f} | "
                    f"elapsed {format_time(elapsed)} | eta {format_time(eta)}"
                )

        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_seen if train_seen else 0.0

        print(f"Epoch {epoch}/{args.epochs} | val   {progress_bar(len(val_loader), len(val_loader))} evaluating")
        val_loss, val_acc, val_bal_acc, val_recalls = evaluate(model, val_loader, criterion, device)
        # The scheduler only reacts to validation loss, not training loss.
        scheduler.step(val_loss)

        # Save only when validation loss makes a real improvement.
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

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Bal Acc: {val_bal_acc:.4f} | "
            f"Recall N/B/S: {val_recalls[0]:.4f}/{val_recalls[1]:.4f}/{val_recalls[2]:.4f} | "
            f"Best Val Loss: {best_val_loss:.4f} @ epoch {best_epoch} | "
            f"status: {status} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"epoch time {format_time(epoch_time)} | total {format_time(total_elapsed)} | "
            f"remaining {format_time(remaining)}"
        )

        if no_improve_count >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best validation loss was {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    # Final test metrics are reported from the best saved checkpoint, not the last epoch.
    model.load_state_dict(torch.load(MODEL_OUTPUT, map_location=device))
    print(f"\nBest Model Test {progress_bar(len(test_loader), len(test_loader))} evaluating")
    test_loss, test_acc, test_bal_acc, test_recalls = evaluate(model, test_loader, criterion, device)
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Bal Acc: {test_bal_acc:.4f} | "
        f"Recall N/B/S: {test_recalls[0]:.4f}/{test_recalls[1]:.4f}/{test_recalls[2]:.4f}"
    )
    print(f"Best model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
