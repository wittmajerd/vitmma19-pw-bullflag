# %%
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

def load_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}

def majority_baseline(y: np.ndarray) -> int:
    vals, counts = np.unique(y, return_counts=True)
    return int(vals[np.argmax(counts)])

def threshold_baseline(X: np.ndarray, thresh: float, bull_cls: int, bear_cls: int, noflag_cls: int, close_idx: int = -1) -> np.ndarray:
    # X: (B, T, F)
    first_close = X[:, 0, close_idx]
    last_close = X[:, -1, close_idx]
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
) -> dict[str, dict[str, float]]:
    arrays = load_arrays(npz_path)
    results: dict[str, dict[str, float]] = {}
    majority_class = majority_baseline(arrays["train_y"])

    for split in ("val", "test"):
        x_key, y_key = f"{split}_X", f"{split}_y"
        if x_key not in arrays or y_key not in arrays or len(arrays[x_key]) == 0:
            continue
        X, y = arrays[x_key], arrays[y_key]

        # Majority baseline
        maj_preds = np.full_like(y, majority_class)
        results.setdefault("majority", {})[split] = {
            "acc": accuracy_score(y, maj_preds),
            "f1_macro": f1_score(y, maj_preds, average="macro"),
        }

        # Threshold baseline
        thr_preds = threshold_baseline(X, thresh, bull_cls, bear_cls, noflag_cls)
        results.setdefault("threshold", {})[split] = {
            "acc": accuracy_score(y, thr_preds),
            "f1_macro": f1_score(y, thr_preds, average="macro"),
        }

    # Print summary
    for name, splits in results.items():
        for split, m in splits.items():
            print(f"{name} | {split}: acc={m['acc']:.3f} f1_macro={m['f1_macro']:.3f}")
    return results

if __name__ == "__main__":
    evaluate_baselines()
