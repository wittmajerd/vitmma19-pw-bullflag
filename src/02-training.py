from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import config
from utils import setup_logger

logger = setup_logger()


class TemporalCNN(nn.Module):
    """1D CNN classifier expecting input (B, T, F)."""
    def __init__(self, input_size: int, num_classes: int, channels: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        in_ch = input_size
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.backbone(x)
        return self.head(x)


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def prepare_dataloaders(arrays: dict[str, np.ndarray], batch_size: int, drop_noflag: bool) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    # Optionally drop No-Flag (0) from train only
    if drop_noflag and "train_y" in arrays:
        mask = arrays["train_y"] != 0
        arrays["train_X"], arrays["train_y"] = arrays["train_X"][mask], arrays["train_y"][mask]
    for split in ("train", "val"):
        x_key, y_key = f"{split}_X", f"{split}_y"
        if x_key in arrays and y_key in arrays and len(arrays[x_key]) > 0:
            loaders[split] = build_loader(arrays[x_key], arrays[y_key], batch_size, shuffle=(split == "train"))
    return loaders

def run_phase(model: nn.Module, loader: DataLoader | None, criterion, device, optimizer=None):
    if loader is None:
        return float("nan"), float("nan"), float("nan")
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    preds_all, targets_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        pred_labels = logits.argmax(1)
        total_correct += (pred_labels == yb).sum().item()
        total_samples += xb.size(0)
        preds_all.append(pred_labels.detach().cpu())
        targets_all.append(yb.detach().cpu())
    preds_concat = torch.cat(preds_all)
    targets_concat = torch.cat(targets_all)
    macro_f1 = f1_score(targets_concat, preds_concat, average="macro")
    return total_loss / total_samples, total_correct / total_samples, macro_f1

def train_and_evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_path = Path(config.DATA_DIR)
    npz_path = data_path / "processed_data.npz"
    arrays = np.load(npz_path)
    n_feats = arrays["train_X"].shape[-1]
    label_arrays = [arrays[k] for k in arrays if k.endswith("_y")]
    num_classes = len(np.unique(np.concatenate(label_arrays)))

    loaders = prepare_dataloaders(
        arrays=arrays,
        batch_size=getattr(config, "BATCH_SIZE", 32),
        drop_noflag=getattr(config, "DROP_NOFLAG_TRAIN", True),
    )

    channels = tuple(getattr(config, "CNN_CHANNELS", (32, 64)))
    dropout = float(getattr(config, "CNN_DROPOUT", 0.1))
    model = TemporalCNN(input_size=n_feats, num_classes=num_classes, channels=channels, dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=getattr(config, "LEARNING_RATE", 1e-3))

    best_val_f1 = 0.0
    patience = getattr(config, "EARLY_STOPPING_PATIENCE", 10)
    wait = 0
    epochs = getattr(config, "EPOCHS", 10)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = run_phase(model, loaders.get("train"), criterion, device, optimizer)
        val_loss, val_acc, val_f1 = run_phase(model, loaders.get("val"), criterion, device)
        logger.info(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )

        if not np.isnan(val_f1) and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), Path(config.MODEL_SAVE_PATH))
            logger.info(f"  Saved best model (val_f1={best_val_f1:.3f}) to {config.MODEL_SAVE_PATH}")
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_and_evaluate()