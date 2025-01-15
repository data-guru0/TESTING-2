import os

########################### DATA INGESTION #########################

RAW_DIR = "artifacts/raw"
CONFIG_PATH = "config/config.yaml"

############################ Data Processing #######################3


PROCESSED_DIR = "artifacts/processed"
ANIMELIST_CSV = "artifacts/raw/animelist.csv"


X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "X_train_array.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "X_test_array.pkl")
Y_TRAIN = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR, "y_test.pkl")

USER2USER_ENCODED = "artifacts/processed/user2user_encoded.pkl"
USER2USER_DECODED = "artifacts/processed/user2user_decoded.pkl"

ANIME2ANIME_ENCODED = "artifacts/processed/anime2anime_encoded.pkl"
ANIME2ANIME_DECODED = "artifacts/processed/anime2anime_decoded.pkl"


########################### MODEL TRAINING ################################3

CHECKPOINT_FILE_PATH = './artifacts/model_checkpoint/weights.weights.h5'
MODEL_DIR = './artifacts/model'
WEIGHTS_DIR = './artifacts/weights'
MODEL_PATH = './artifacts/model/model.h5'
WEIGHTS_PATH_USER = './artifacts/weights/user_weights.pkl'
WEIGHTS_PATH_ANIME = './artifacts/weights/anime_weights.pkl'


################################## UI APP ##########################333

RATING_DF = "artifacts/processed/rating_df.csv"
DF_PATH = "artifacts/processed/anime_df.csv"
DF_SYNOPSIS = "artifacts/processed/synopsis_df.csv"

USER_WEIGHTS = "artifacts/weights/user_weights.pkl"
ANIME_WEIGHTS = "artifacts/weights/anime_weights.pkl"