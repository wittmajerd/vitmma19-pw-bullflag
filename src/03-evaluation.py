import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import config
from utils import setup_logger

logger = setup_logger()

# -----------------------------------------------------------------------------
# Model (same as in training) - cant we import from training?
from importlib import import_module
train_module = import_module('02-training')
TemporalCNN = train_module.TemporalCNN

# class TemporalCNN(nn.Module):
#     """1D CNN classifier expecting input (B, T, F)."""
#     def __init__(self, input_size: int, num_classes: int, channels: tuple[int, ...], dropout: float):
#         super().__init__()
#         layers = []
#         in_ch = input_size
#         for out_ch in channels:
#             layers += [
#                 nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
#                 nn.BatchNorm1d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout),
#             ]
#             in_ch = out_ch
#         self.backbone = nn.Sequential(*layers)
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             nn.Linear(in_ch, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.transpose(1, 2)  # (B, F, T)
#         x = self.backbone(x)
#         return self.head(x)

# -----------------------------------------------------------------------------
def load_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}

def majority_baseline(y: np.ndarray) -> int:
    vals, counts = np.unique(y, return_counts=True)
    return int(vals[np.argmax(counts)])

def threshold_baseline(
    X: np.ndarray,
    thresh: float,
    bull_cls: int,
    bear_cls: int,
    noflag_cls: int,
    close_idx: int = -1,
    lookback: int = 4,
    offset: int = 0,
) -> np.ndarray:
    T = X.shape[1]
    lookback = max(1, lookback)
    last_idx = max(0, T - 1 - offset)
    start_idx = max(0, last_idx - (lookback - 1))
    first_close = X[:, start_idx, close_idx]
    last_close = X[:, last_idx, close_idx]
    ret = (last_close - first_close) / np.clip(np.abs(first_close), 1e-8, None)
    preds = np.full(len(ret), noflag_cls, dtype=int)
    preds[ret > thresh] = bull_cls
    preds[ret < -thresh] = bear_cls
    return preds

def evaluate_baselines(
    npz_path: Path,
    thresh: float = 0.01,
    bull_cls: int = 1,
    bear_cls: int = 4,
    noflag_cls: int = 0,
    lookback: int = 4,
    offset: int = 0,
) -> Dict[str, Dict[str, float]]:
    arrays = load_arrays(npz_path)
    results: Dict[str, Dict[str, float]] = {}
    majority_class = majority_baseline(arrays["train_y"])

    for split in ("val", "test"):
        x_key, y_key = f"{split}_X", f"{split}_y"
        if x_key not in arrays or y_key not in arrays or len(arrays[x_key]) == 0:
            continue
        X, y = arrays[x_key], arrays[y_key]

        maj_preds = np.full_like(y, majority_class)
        results.setdefault("majority", {})[split] = {
            "acc": accuracy_score(y, maj_preds),
            "f1_macro": f1_score(y, maj_preds, average="macro"),
        }

        thr_preds = threshold_baseline(X, thresh, bull_cls, bear_cls, noflag_cls, lookback=lookback, offset=offset)
        results.setdefault("threshold", {})[split] = {
            "acc": accuracy_score(y, thr_preds),
            "f1_macro": f1_score(y, thr_preds, average="macro"),
        }

    for name, splits in results.items():
        for split, m in splits.items():
            logger.info("%s | %s: acc=%.3f f1_macro=%.3f", name, split, m["acc"], m["f1_macro"])
    return results

# -----------------------------------------------------------------------------
def load_model(arrays: Dict[str, np.ndarray], model_path: Path, device: torch.device) -> TemporalCNN:
    n_feats = arrays["train_X"].shape[-1]
    label_arrays = [arrays[k] for k in arrays if k.endswith("_y")]
    num_classes = len(np.unique(np.concatenate(label_arrays)))
    channels = tuple(getattr(config, "CNN_CHANNELS", (32, 64)))
    dropout = float(getattr(config, "CNN_DROPOUT", 0.1))
    model = TemporalCNN(n_feats, num_classes, channels, dropout).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def evaluate_model(
    npz_path: Path,
    model_path: Path,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    device_t = torch.device(device)
    arrays = load_arrays(npz_path)
    model = load_model(arrays, model_path, device_t)
    results: Dict[str, Dict[str, float]] = {}

    for split in ("val", "test"):
        x_key, y_key = f"{split}_X", f"{split}_y"
        if x_key not in arrays or y_key not in arrays or len(arrays[x_key]) == 0:
            continue
        X, y = arrays[x_key], arrays[y_key]
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device_t)
        with torch.no_grad():
            logits = model(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        results.setdefault("temporal_cnn", {})[split] = {
            "acc": accuracy_score(y, preds),
            "f1_macro": f1_score(y, preds, average="macro"),
        }

    for split, m in results.get("temporal_cnn", {}).items():
        logger.info("temporal_cnn | %s: acc=%.3f f1_macro=%.3f", split, m["acc"], m["f1_macro"])
    return results

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = Path(config.DATA_DIR) / "processed_data.npz"
    model_path = Path(config.MODEL_SAVE_PATH)

    logger.info("Running baselines on %s", data_path)
    evaluate_baselines(npz_path=data_path)

    if model_path.exists():
        logger.info("Evaluating trained model from %s", model_path)
        evaluate_model(npz_path=data_path, model_path=model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        logger.info("Model path %s not found; skipping model evaluation.", model_path)