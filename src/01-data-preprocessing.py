import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

import config
from utils import setup_logger

logger = setup_logger()

def normalize_timestamp(timestamp):
    try:
        # Case 1: Check if it's a Unix timestamp in milliseconds or seconds
        if isinstance(timestamp, (int, float)) or (isinstance(timestamp, str) and timestamp.isdigit()):
            timestamp = int(timestamp)  # Ensure it's an integer
            if len(str(timestamp)) == 13:  # Milliseconds
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            elif len(str(timestamp)) == 10:  # Seconds
                return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        # Case 2: Check if it's already in a recognizable datetime format
        elif isinstance(timestamp, str):
            # Try parsing with known formats
            for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(timestamp, fmt).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue

        # If none of the above cases match, raise an error
        raise ValueError(f"Unrecognized timestamp format: {timestamp}")

    except Exception as e:
        logger.info(f"Error normalizing timestamp: {e}")
        return None

def normalize_data(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    - standardize column names (lower/strip)
    - convert timestamp column to UTC datetime
    - sort rows by timestamp and reset the index
    - drops volume column if exists
    """
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    timestamp_candidates = ["timestamp", "time", "date"]
    timestamp_col = next((c for c in timestamp_candidates if c in df.columns), None)
    if not timestamp_col:
        raise ValueError("No timestamp-like column found in dataframe.")

    ts_normalized = df[timestamp_col].apply(normalize_timestamp)
    df[timestamp_col] = pd.to_datetime(ts_normalized, utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(by=timestamp_col).reset_index(drop=True)

    if timestamp_col != "timestamp":
        df = df.rename(columns={timestamp_col: "timestamp"})

    if "volume" in df.columns:
        df = df.drop(columns=["volume"])

    return df

def parse_labels(json_file: str, folder_path: str) -> dict[str, List[Dict[str, any]]]:
    """Load labels from a JSON export, arrange them in a dictionary and normalize timestamps per file."""
    with open(os.path.join(folder_path, json_file), 'r', encoding='utf-8') as file:
        data = json.load(file)

    folder_dict = {}

    for item in data:
        # Determine format
        if 'annotations' in item:
            file_name = item['data']['csv']
            # Remove hash: split by '-' and take from second part
            file_name_clean = '-'.join(file_name.split('-')[1:]) if '-' in file_name else file_name
            for annotation in item['annotations']:
                for result in annotation['result']:
                    label_list = result['value']['timeserieslabels']
                    # Convert list to string (take first if single, or join)
                    label = label_list[0] if len(label_list) == 1 else ' '.join(label_list)
                    start_time = result['value']['start']
                    end_time = result['value']['end']
                    if file_name_clean not in folder_dict:
                        folder_dict[file_name_clean] = []
                    folder_dict[file_name_clean].append({
                        'label': label,
                        'start': pd.to_datetime(normalize_timestamp(start_time), utc=True),
                        'end': pd.to_datetime(normalize_timestamp(end_time), utc=True)
                    })
        elif 'label' in item:
            file_name = item['csv']
            # Remove hash: split by '-' and take from second part
            file_name_clean = '-'.join(file_name.split('-')[1:]) if '-' in file_name else file_name
            for label_entry in item['label']:
                label_list = label_entry['timeserieslabels']
                # Convert list to string (take first if single, or join)
                label = label_list[0] if len(label_list) == 1 else ' '.join(label_list)
                start_time = label_entry['start']
                end_time = label_entry['end']
                if file_name_clean not in folder_dict:
                    folder_dict[file_name_clean] = []
                folder_dict[file_name_clean].append({
                    'label': label,
                    'start': pd.to_datetime(normalize_timestamp(start_time), utc=True),
                    'end': pd.to_datetime(normalize_timestamp(end_time), utc=True)
                })
        else:
            logger.info(f"Unknown format for item ID: {item.get('id', 'N/A')}")

    return folder_dict

def get_parsed_labels(data_path: str, folders: List[str]) -> dict[str, dict[str, List[Dict[str, any]]]]:
    """ Parse labels from all folders and return a nested dictionary."""
    logger.info("Parsing labels ...")
    all_labels = {}
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

            if not json_files:
                logger.info(f'No JSON file found in folder: {folder}')
                continue
            logger.info(f'Folder: {folder}, JSON files: {json_files}')

            folder_dict = {}
            for json_file in json_files:
                parsed_labels = parse_labels(json_file, folder_path)
                for file_name, labels in parsed_labels.items():
                    if file_name not in folder_dict:
                        folder_dict[file_name] = []
                    folder_dict[file_name].extend(labels)

            # Deduplicate labels within each file
            for file_name, labels in folder_dict.items():
                unique_labels = {frozenset(label.items()): label for label in labels}.values()
                folder_dict[file_name] = list(unique_labels)

            # Add the folder's data to the main dictionary
            all_labels[folder] = folder_dict
    logger.info("Finished parsing labels.")
    return all_labels

def get_data_with_labels(data_path: str, all_labels: dict[str, dict[str, List[Dict[str, any]]]]) -> dict[str, dict[str, any]]:
    data_with_labels = {}
    logger.info("Loading data ...")
    for folder, files in all_labels.items():
        folder_path = os.path.join(data_path, folder)
        for file_name, labels in files.items():
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                data = normalize_data(data)
                # Attach the labels to the data
                key = f"{folder}/{file_name}"
                data_with_labels[key] = {
                    'data': data,
                    'labels': labels
                }
    logger.info("Finished loading data.")
    return data_with_labels

def label_ohlc_df(
    ohlc_df: pd.DataFrame,
    labels: List[Dict],
    label_map: Dict[str, int],
    timestamp_col: str = "timestamp",
    target_col: str = "flag_label"
) -> pd.DataFrame:
    """
    Sor-szintű címkézés: minden sornak megadja a hozzá tartozó flag ID-t.
    Feltételezi, hogy a DataFrame-ben van timestamp oszlop.
    """
    df = ohlc_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df[target_col] = label_map.get("No Flag", 0)

    for entry in labels:
        start = pd.to_datetime(entry["start"], utc=True)
        end = pd.to_datetime(entry["end"], utc=True)
        cls_id = label_map.get(entry["label"], label_map.get("No Flag", 0))
        mask = (df[timestamp_col] >= start) & (df[timestamp_col] <= end)
        df.loc[mask, target_col] = cls_id

    return df

def trim_after_last_label(df: pd.DataFrame, label_col: str = "flag_label", margin: int = 64, label_map: dict[str, int] = None) -> pd.DataFrame:
    if df.shape[0] < 10000:
        return df
    
    mask = df[label_col] != label_map["No Flag"]
    last_labeled_idx = mask.to_numpy().nonzero()[0][-1]
    end_idx = min(len(df), last_labeled_idx + margin)
    return df.iloc[:end_idx].reset_index(drop=True)

def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    timestamp_col: str = "timestamp"
) -> dict[str, pd.DataFrame]:
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": df_sorted.iloc[:train_end].copy(),
        "val": df_sorted.iloc[train_end:val_end].copy(),
        "test": df_sorted.iloc[val_end:].copy(),
    }

def generate_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 64,
    step: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    features = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy()
    X, y = [], []
    for end_idx in range(window_size, len(df), step):
        start_idx = end_idx - window_size
        window = features[start_idx:end_idx]
        target = targets[end_idx - 1]
        X.append(window)
        y.append(target)
    return np.stack(X), np.array(y)

def normalize_splits(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str]
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    scaler = StandardScaler()
    train_features = splits["train"][feature_cols]
    scaler.fit(train_features)
    normalized = {}
    for key, split_df in splits.items():
        norm_df = split_df.copy()
        norm_df[feature_cols] = scaler.transform(split_df[feature_cols])
        normalized[key] = norm_df
    return normalized, scaler

def process_data(
        data_with_labels,
        window_size = 32,
        step = 1,
        train_ratio = 0.7,
        val_ratio = 0.15
    ):
    logger.info("Processing data ...")
    
    window_arrays = {"train_X": [],"train_y": [],"val_X": [],"val_y": [],"test_X": [],"test_y": [],}

    for file_name, content in data_with_labels.items():
        data = content['data']
        labels = content['labels']

        # extended_labels = extend_labels_with_pole(data, labels)
        # logger.info(f"File: {file_name}, Original labels: {len(labels)}, Extended labels: {len(extended_labels)}")

        labeled_df = label_ohlc_df(data, labels, config.LABEL_MAP)

        if len(labeled_df["flag_label"].unique()) < 2 and labeled_df["flag_label"].unique()[0] == 0:
            logger.info(f"Skip {file_name}: no flag labels present")
            continue

        labeled_df = trim_after_last_label(labeled_df, margin=window_size, label_map=config.LABEL_MAP)

        df_len = len(labeled_df)

        if df_len < window_size:
            logger.info(f"Skip {file_name}: insufficient rows ({df_len}) for window_size={window_size}")
            continue

        if df_len * min(train_ratio, val_ratio, (1 - train_ratio - val_ratio)) < window_size:
            # skip file splitting use all for training
            splits = {"train": labeled_df}
        else:
            splits = split_time_series(labeled_df, train_ratio=train_ratio, val_ratio=val_ratio)

        normalized_splits, _ = normalize_splits(splits, feature_cols=config.FEATURES)

        for split_name, split_df in normalized_splits.items():
            if len(split_df) < window_size:
                logger.info(f"Skip {split_name}: insufficient rows ({len(split_df)}) for window_size={window_size}")
                continue
            X, y = generate_windows(split_df, feature_cols=config.FEATURES, target_col="flag_label", window_size=window_size, step=step)
            
            window_arrays[f"{split_name}_X"].append(X)
            window_arrays[f"{split_name}_y"].append(y)

    logger.info("Finished processing data.")
    return window_arrays

def save_processed_data(
        window_arrays: dict[str, List[np.ndarray]],
        data_path: Path,
    ):
    stacked_train_X = np.concatenate(window_arrays["train_X"], axis=0)
    stacked_train_y = np.concatenate(window_arrays["train_y"], axis=0)
    logger.info(f"Train shape: {stacked_train_X.shape}, {stacked_train_y.shape}")

    stacked_val_X = np.concatenate(window_arrays["val_X"], axis=0)
    stacked_val_y = np.concatenate(window_arrays["val_y"], axis=0)
    logger.info(f"Validation shape: {stacked_val_X.shape}, {stacked_val_y.shape}")

    stacked_test_X = np.concatenate(window_arrays["test_X"], axis=0)
    stacked_test_y = np.concatenate(window_arrays["test_y"], axis=0)
    logger.info(f"Test shape: {stacked_test_X.shape}, {stacked_test_y.shape}")

    np.savez_compressed(
        data_path / "processed_train_data.npz",
        train_X=stacked_train_X,
        train_y=stacked_train_y,
        val_X=stacked_val_X,
        val_y=stacked_val_y,
    )

    np.savez_compressed(
        data_path / "processed_test_data.npz",
        test_X=stacked_test_X,
        test_y=stacked_test_y,
    )
    logger.info(f"Processed train and test data saved to {data_path}")

def download_and_extract_data(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading data to {out_path} ...")

    with requests.get(config.URL, stream=True) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    logger.info(f"Extracting data to {out_path.parent} ...")
    with ZipFile(out_path, "r") as archive:
        archive.extractall(out_path.parent)


if __name__ == "__main__":
    parent_path = Path(__file__).parent.parent
    data_path = parent_path / "data"
    output_path = parent_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    download_and_extract_data(data_path / "data.zip")

    data_path = data_path / "bullflagdetector"
    folders = os.listdir(data_path)
    if 'consensus' in folders:
        folders.remove('consensus')
    if 'sample' in folders:
        folders.remove('sample')

    labels = get_parsed_labels(data_path, folders)
    data_with_labels = get_data_with_labels(data_path, labels)
    
    processed_data = process_data(data_with_labels)
    save_processed_data(processed_data, output_path, filename="processed_data.npz")
