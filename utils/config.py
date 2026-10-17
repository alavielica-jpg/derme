from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"            # data/raw/<class_name>/*.jpg
PROC_DIR = ROOT / "data" / "processed"     # train/ val/ test/ created by organize_data.py
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DB_PATH = ROOT / "screener.db"

MODEL_PATH = MODEL_DIR / "skin_model.keras"
META_PATH = MODEL_DIR / "meta.json"            # class names, backbone, img size
THRESHOLD_PATH = MODEL_DIR / "threshold.json"  # tuned by evaluate.py

BACKBONE = "mobilenetv2"  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LAST_N_LAYERS = 30
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5

MIN_SIDE_PX = 451            # resolution floor (shortest side)
BLUR_THRESHOLD = 15.0       # Laplacian variance below this = blurry
DARK_MEAN = 50               # mean gray below = too dark
BRIGHT_MEAN = 190            # mean gray above = too bright
MAX_CLIPPED_DARK_FRAC = 0.40    # fraction of near-black pixels
MAX_CLIPPED_BRIGHT_FRAC = 0.30  # fraction of blown-out pixels
MIN_SKIN_FRACTION = 0.20     # below this: "no skin detected"

DEFAULT_CONF_THRESHOLD = 0.60  # below = flagged "inconclusive"
