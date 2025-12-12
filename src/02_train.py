import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

    def save_model(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str):
        self.load_state_dict(torch.load(filepath))
        self.eval()


class TemporalCNN(nn.Module):
    """1D CNN with residual blocks for sequence windows shaped (B, T, F)."""
    def __init__(self, input_channels: int, num_classes: int, channels: tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        layers = []
        in_ch = input_channels
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
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


class BiGRUClassifier(nn.Module):
    """Bidirectional GRU using final hidden state."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
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
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # concat directions
        return self.fc(h_last)


class TransformerClassifier(nn.Module):
    """Lightweight Transformer encoder pooling CLS token."""
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
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, self.input_proj(x)], dim=1)
        enc = self.encoder(x)
        return self.head(enc[:, 0])

# %%
def load_processed_dataset(npz_file: Path) -> dict[str, np.ndarray]:
    if not npz_file.exists():
        raise FileNotFoundError(f"Missing file: {npz_file}")
    with np.load(npz_file) as data:
        return {key: data[key] for key in data.files}

def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def prepare_dataloaders(arrays: dict[str, np.ndarray], batch_size: int) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        x_key, y_key = f"{split}_X", f"{split}_y"
        if x_key in arrays and y_key in arrays and len(arrays[x_key]) > 0:
            loaders[split] = build_loader(arrays[x_key], arrays[y_key], batch_size, shuffle=(split == "train"))
    return loaders

def run_phase(model, loader, criterion, device, optimizer=None):
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

def train_and_validate(model_builder, config: dict = None, processed_file: Path = Path("data/processed_dataset.npz")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays = load_processed_dataset(processed_file)
    n_features = arrays["train_X"].shape[-1]
    label_arrays = [arrays[key] for key in arrays if key.endswith("_y")]
    classes = np.unique(np.concatenate(label_arrays))
    num_classes = len(classes)

    loaders = prepare_dataloaders(arrays, config["batch_size"])

    model = model_builder(config, n_features, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_f1 = 0.0
    for epoch in range(1, config["num_epochs"] + 1):
        train_loss, train_acc, train_f1 = run_phase(model, loaders.get("train"), criterion, device, optimizer)
        val_loss, val_acc, val_f1 = run_phase(model, loaders.get("val"), criterion, device)

        msg = (f"Epoch {epoch:02d} | "
               f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")
        print(msg)

        if not np.isnan(val_f1) and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")

    if "test" in loaders:
        test_loss, test_acc, test_f1 = run_phase(model, loaders["test"], criterion, device)
        print(f"Test  | loss={test_loss:.4f} acc={test_acc:.3f} f1={test_f1:.3f}")


MODEL_REGISTRY = {
    "lstm": lambda cfg, n_feats, n_classes: LSTMClassifier(
        input_size=n_feats,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        num_classes=n_classes,
    ),
    "temporal_cnn": lambda cfg, n_feats, n_classes: TemporalCNN(n_feats, n_classes),
    "bigru": lambda cfg, n_feats, n_classes: BiGRUClassifier(
        input_size=n_feats,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        num_classes=n_classes,
        dropout=cfg.get("dropout", 0.2),
    ),
    "transformer": lambda cfg, n_feats, n_classes: TransformerClassifier(
        input_size=n_feats,
        num_classes=n_classes,
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 256),
        dropout=cfg.get("dropout", 0.1),
    ),
}

config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10
}

train_and_validate(MODEL_REGISTRY[ "temporal_cnn"], config, Path("data/processed_dataset.npz"))