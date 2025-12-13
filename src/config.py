# Training hyperparameters
EPOCHS = 1
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DROP_NOFLAG_TRAIN = True  # drop label 0 samples from train split

# Model (Temporal CNN)
CNN_CHANNELS = (32, 64)
CNN_DROPOUT = 0.1

# Paths
DATA_DIR = "/app/data"
MODEL_SAVE_PATH = "/app/model.pth"

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


# maradjunk a cnn-nél esetleg lstm de cnn is lehet sokféle kéne egy optimálisabbat találni ami nagyon gyors
# lehet window size lehetne kissebb nagyobb?

# ez majd visszaolvad a training fileba mert elég 1-2 modell nem kell külön file

# CHECKLIST:
# adat letöltés automatikusan kicsomagolás - feldolgozásban jó
# https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAlEFc87da4SLpRVTCs81KwATOAjf5GzI-IxEED_nGrjh0?e=eGgGec&download=1

# adat kimentés külön a teszteléshez
# eval baseline + savát legjobb visszatöltés

# config fájl használata
# logger használata print helyett

# notebook tisztogatás
# kommentek rendezése
# repo rendberakás, pathek, conténerezés