# %%
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

import config
from utils import setup_logger

logger = setup_logger()

from importlib import import_module
train_module = import_module('02-training')
TemporalCNN = train_module.TemporalCNN

def log_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    label_name_map: Dict[int, str] | None = None,
    save_dir: Path | None = None,
    normalize: str | None = "true",
) -> None:
    """Log per-class report and save a confusion matrix."""
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [label_name_map.get(l, str(l)) for l in labels] if label_name_map else [str(l) for l in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=3,
        zero_division=0.0,  # show the warning; don’t drop results
    )
    logger.info("%s classification report:\n%s", name, report)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto", origin="upper")
    ax.set_title(f"{name} confusion matrix (normalized)")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(target_names)
    ax.xaxis.set_ticks_position("bottom")  # put x-labels at bottom
    fig.tight_layout()
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"cm_{name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("%s confusion matrix saved to %s", name, out_path)
    plt.close(fig)

def load_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}

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
    majority_class = 0
    label_name_map = {v: k for k, v in getattr(config, "LABEL_MAP", {}).items()}
    save_dir = Path(getattr(config, "DATA_DIR", "output"))

    split = "test"
    x_key, y_key = f"{split}_X", f"{split}_y"
    if x_key in arrays and y_key in arrays and len(arrays[x_key]) > 0:
        X, y = arrays[x_key], arrays[y_key]

        maj_preds = np.full_like(y, majority_class)
        results.setdefault("majority", {})[split] = {
            "acc": accuracy_score(y, maj_preds),
            "f1_macro": f1_score(y, maj_preds, average="macro"),
        }
        log_diagnostics(y, maj_preds, name="majority", label_name_map=label_name_map, save_dir=save_dir)

        thr_preds = threshold_baseline(X, thresh, bull_cls, bear_cls, noflag_cls, lookback=lookback, offset=offset)
        results.setdefault("threshold", {})[split] = {
            "acc": accuracy_score(y, thr_preds),
            "f1_macro": f1_score(y, thr_preds, average="macro"),
        }
        log_diagnostics(y, thr_preds, name="threshold", label_name_map=label_name_map, save_dir=save_dir)

    return results

# -----------------------------------------------------------------------------
def load_model(arrays: Dict[str, np.ndarray], model_path: Path, device: torch.device) -> TemporalCNN:
    n_feats = arrays["test_X"].shape[-1]
    label_arrays = [arrays[k] for k in arrays if k.endswith("_y")]
    num_classes = len(np.unique(np.concatenate(label_arrays)))
    channels = tuple(getattr(config, "CNN_CHANNELS", (32, 64)))
    dropout = float(getattr(config, "CNN_DROPOUT", 0.1))
    model = TemporalCNN(n_feats, num_classes, channels, dropout).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:  # older torch without weights_only
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
    label_name_map = {v: k for k, v in getattr(config, "LABEL_MAP", {}).items()}
    save_dir = Path(getattr(config, "DATA_DIR", "output"))

    split = "test"
    x_key, y_key = f"{split}_X", f"{split}_y"
    if x_key in arrays and y_key in arrays and len(arrays[x_key]) > 0:
        X, y = arrays[x_key], arrays[y_key]
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device_t)
        with torch.no_grad():
            logits = model(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        results.setdefault("temporal_cnn", {})[split] = {
            "acc": accuracy_score(y, preds),
            "f1_macro": f1_score(y, preds, average="macro"),
        }
        log_diagnostics(y, preds, name="temporal_cnn", label_name_map=label_name_map, save_dir=save_dir)

    return results

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = Path(config.DATA_DIR) / "processed_test_data.npz"
    model_path = Path(config.MODEL_SAVE_PATH)

    logger.info("Running baselines on %s", data_path)
    base_results = evaluate_baselines(npz_path=data_path)
    # logger.info("Baseline results: %s", base_results)

    if model_path.exists():
        logger.info("Evaluating trained model from %s", model_path)
        model_results = evaluate_model(npz_path=data_path, model_path=model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        # logger.info("Model results: %s", model_results)
    else:
        logger.info("Model path %s not found; skipping model evaluation.", model_path)