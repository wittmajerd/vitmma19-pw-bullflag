# %%
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import torch
# from 02_train import LSTMClassifier, TemporalCNN, BiGRUClassifier, TransformerClassifier

from utils import setup_logger

logger = setup_logger()

def load_arrays(npz_path: Path) -> dict[str, np.ndarray]:
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
    npz_path: Path = Path(__file__).parent.parent / "data" / "processed_data.npz",
    thresh: float = 0.01,
    bull_cls: int = 1,
    bear_cls: int = 4,
    noflag_cls: int = 0,
    lookback: int = 4,
    offset: int = 0,
) -> dict[str, dict[str, float]]:
    arrays = load_arrays(npz_path)
    results: dict[str, dict[str, float]] = {}
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
            print(f"{name} | {split}: acc={m['acc']:.3f} f1_macro={m['f1_macro']:.3f}")
    return results

def load_trained_model(model_class, model_path: Path, device: str = "cpu"):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_trained_models(
    npz_path: Path = Path(__file__).parent.parent / "data" / "processed_data.npz",
    model_paths: dict[str, Path] = {},
    device: str = "cpu",
) -> dict[str, dict[str, float]]:
    arrays = load_arrays(npz_path)
    results: dict[str, dict[str, float]] = {}

    model_classes = {
        "lstm": LSTMClassifier,
        "temporal_cnn": TemporalCNN,
        "bigru": BiGRUClassifier,
        "transformer": TransformerClassifier,
    }

    for model_name, model_path in model_paths.items():
        if model_name not in model_classes:
            print(f"Unknown model name: {model_name}, skipping.")
            continue
        model = load_trained_model(model_classes[model_name], model_path, device)

        for split in ("val", "test"):
            x_key, y_key = f"{split}_X", f"{split}_y"
            if x_key not in arrays or y_key not in arrays or len(arrays[x_key]) == 0:
                continue
            X, y = arrays[x_key], arrays[y_key]
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(X_tensor)
                preds = logits.argmax(dim=1).cpu().numpy()

            results.setdefault(model_name, {})[split] = {
                "acc": accuracy_score(y, preds),
                "f1_macro": f1_score(y, preds, average="macro"),
            }

    for name, splits in results.items():
        for split, m in splits.items():
            print(f"{name} | {split}: acc={m['acc']:.3f} f1_macro={m['f1_macro']:.3f}")
    return results


if __name__ == "__main__":
    evaluate_baselines()
    model_paths = {
        "lstm": Path(__file__).parent.parent / "models" / "lstm_model.pth",
        "temporal_cnn": Path(__file__).parent.parent / "models" / "temporal_cnn_model.pth",
        "bigru": Path(__file__).parent.parent / "models" / "bigru_model.pth",
        "transformer": Path(__file__).parent.parent / "models" / "transformer_model.pth",
    }
    evaluate_trained_models(model_paths=model_paths)
    # printing results
