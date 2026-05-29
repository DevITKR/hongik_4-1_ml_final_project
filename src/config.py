from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RAVDESS"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

METADATA_PATH = PROCESSED_DIR / "metadata.csv"
TRAIN_SPLIT_PATH = SPLITS_DIR / "train.csv"
VAL_SPLIT_PATH = SPLITS_DIR / "val.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test.csv"
BEST_MODEL_PATH = SAVED_MODELS_DIR / "wav2vec2_bilstm_attention_best.pth"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

RANDOM_SEED = 42

TRAIN_ACTORS = tuple(range(1, 19))
VAL_ACTORS = tuple(range(19, 22))
TEST_ACTORS = tuple(range(22, 25))

LABEL_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4,
    "disgust": 5,
    "surprised": 6,
}

RAVDESS_EMOTION_ID_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

METADATA_COLUMNS = ["path", "filename", "emotion", "label", "actor"]

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 4
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS

WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
WAV2VEC2_HIDDEN_SIZE = 768
WAV2VEC2_NUM_HIDDEN_STATES = 13
FREEZE_WAV2VEC2 = True
USE_WEIGHTED_LAYER_SUM = True

NUM_CLASSES = len(LABEL_MAP)
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
ATTENTION_NUM_HEADS = 4
ATTENTION_DROPOUT = 0.3
CLASSIFIER_HIDDEN_SIZE = 256
CLASSIFIER_DROPOUT = 0.3

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
