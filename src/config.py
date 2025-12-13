# Configuration settings as an example

# Training hyperparameters
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Paths
DATA_DIR = "/app/data"
MODEL_SAVE_PATH = "/app/model.pth"

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