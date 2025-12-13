from pathlib import Path

# Training hyperparameters
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DROP_NOFLAG_TRAIN = True   # drop label 0 samples from train split
NOFLAG_KEEP = 100_000         # if dropping, keep this many noflag samples in train split

# Model (Temporal CNN)
CNN_CHANNELS = (32, 64)
CNN_DROPOUT = 0.1

# Paths
# DATA_DIR = "/app/data"
# MODEL_SAVE_PATH = "/app/model.pth"
parent_path = Path(__file__).parent.parent
DATA_DIR = parent_path / "output"
MODEL_SAVE_PATH = DATA_DIR / "model.pth"

# Data source
URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAlEFc87da4SLpRVTCs81KwATOAjf5GzI-IxEED_nGrjh0?e=eGgGec&download=1"

LABEL_MAP = {
        "No Flag": 0,
        "Bullish Normal": 1,
        "Bullish Wedge": 2,
        "Bullish Pennant": 3,
        "Bearish Normal": 4,
        "Bearish Wedge": 5,
        "Bearish Pennant": 6,
    }

FEATURES = ['open', 'high', 'low', 'close']
WINDOW_SIZE = 32
STEP = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15



# adat kimentés külön a teszteléshez
# eval baseline + savát legjobb visszatöltés

# label nélküli window eldobás a trainből legyen opció és esetleg aránnyal random shuffle-ölve

# config fájl használata
# logger használata print helyett

# notebook tisztogatás
# kommentek rendezése
# repo rendberakás, pathek, conténerezés