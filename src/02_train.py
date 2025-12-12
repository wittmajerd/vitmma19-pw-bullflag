# %%
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from typing import Callable, Dict, Any




class LSTMClassifier(nn.Module):
    """LSTM classifier expecting input (B, T, F)."""
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = x.new_zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = x.new_zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

    def save_model(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str):
        self.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.eval()


class TemporalCNN(nn.Module):
    """1D CNN classifier expecting input (B, T, F)."""
    def __init__(self, input_size: int, num_classes: int, channels: tuple[int, ...] = (64, 128, 256), dropout: float = 0.2):
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

    def save_model(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str):
        self.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.eval()


class BiGRUClassifier(nn.Module):
    """Bidirectional GRU classifier expecting input (B, T, F)."""
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h_last)

    def save_model(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str):
        self.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.eval()


class TransformerClassifier(nn.Module):
    """Lightweight Transformer encoder pooling CLS token; input (B, T, F)."""
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, self.input_proj(x)], dim=1)
        enc = self.encoder(x)
        return self.head(enc[:, 0])

    def save_model(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str):
        self.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.eval()

# %%

def build_model(config: dict, n_feats: int, n_classes: int) -> nn.Module:
    model_name = config.get("model_name", "temporal_cnn")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}")
    model_params = config.get("model_params", {})
    return MODEL_REGISTRY[model_name](
        model_params if isinstance(model_params, dict) else config,
        n_feats,
        n_classes,
    )

def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def prepare_dataloaders(arrays: dict[str, np.ndarray], batch_size: int) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
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

def train_and_evaluate(config: dict, processed_file: Path = Path("data/processed_dataset.npz")) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = np.load(processed_file)
    n_feats = arrays["train_X"].shape[-1]
    label_arrays = [arrays[k] for k in arrays if k.endswith("_y")]
    num_classes = len(np.unique(np.concatenate(label_arrays)))

    loaders = prepare_dataloaders(arrays, config.get("batch_size", 32))
    model = build_model(config, n_feats, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))

    best_val_f1 = 0.0
    epochs = config.get("num_epochs", 10)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = run_phase(model, loaders.get("train"), criterion, device, optimizer)
        val_loss, val_acc, val_f1 = run_phase(model, loaders.get("val"), criterion, device)
        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")
        if not np.isnan(val_f1) and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_name = config.get("model_name", "temporal_cnn")
            torch.save(model.state_dict(), f"best_model_{model_name}.pt")

    if "test" in loaders:
        test_loss, test_acc, test_f1 = run_phase(model, loaders["test"], criterion, device)
        print(f"Test  | loss={test_loss:.4f} acc={test_acc:.3f} f1={test_f1:.3f}")



MODEL_REGISTRY = {
    "lstm": lambda cfg, n_feats, n_classes: LSTMClassifier(
        input_size=n_feats,
        num_classes=n_classes,
        **cfg,
    ),
    "temporal_cnn": lambda cfg, n_feats, n_classes: TemporalCNN(
        input_size=n_feats,
        num_classes=n_classes,
        **cfg,
    ),
    "bigru": lambda cfg, n_feats, n_classes: BiGRUClassifier(
        input_size=n_feats,
        num_classes=n_classes,
        **cfg,
    ),
    "transformer": lambda cfg, n_feats, n_classes: TransformerClassifier(
        input_size=n_feats,
        num_classes=n_classes,
        **cfg,
    ),
}

# Example config to drive everything
if __name__ == "__main__":
    cfg = {
        "model_name": "temporal_cnn",      # one of: lstm, temporal_cnn, bigru, transformer
        "model_params": {                  # passed into MODEL_REGISTRY builder
            # "hidden_size": 128,
            # "num_layers": 2,
            "dropout": 0.5,
        },
        "batch_size": 32,
        "learning_rate": 0.01,
        "num_epochs": 1,
    }
    data_path = Path(__file__).parent.parent / "data"
    train_and_evaluate(cfg, data_path / "processed_data2.npz")
