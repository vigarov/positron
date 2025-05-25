from .metric import Metric
import numpy as np
from pathlib import Path
import os

# Get project root (adjust as needed based on your folder structure)
NN_ROOT = Path(__file__).parent.parent


DATA_ROOT = NN_ROOT / "data"
IN_PATH = DATA_ROOT / "datasets" / "wb"
OUT_PATH = DATA_ROOT / "checkpoints"

# Various paths
DATA_PATH = IN_PATH / "wb_color_checkers.csv"
GRAYS_DATA = IN_PATH / "wb_grays.csv"
MODEL_BASE_DIR = OUT_PATH / "models"

# Default image paths
DEFAULT_SOURCE_IMG = DATA_ROOT / "prepro" / "cropped" / "source1" / "DSCF7393.tif"
DEFAULT_GT_IMG = DATA_ROOT / "prepro" / "da_tiff" / "DSCF7041.tif"

# Default checkpoint paths
DEFAULT_MODEL_NAME = "enhcombgr3_tanh_hidden_100_10_true_reduce_on_plateau"
DEFAULT_METRIC = "D_LAB"
DEFAULT_CHECKPOINT_PATH = MODEL_BASE_DIR / DEFAULT_MODEL_NAME / DEFAULT_METRIC

# Default evaluation paths
DEFAULT_EVAL_PATH = "./outs"
DEFAULT_NUM_SAMPLES = 5

# Reproduction
REPRO_SEED = 42
TEXT_SEPARATOR = "-"*80

# Metrics
def rmse(gt,preds):
    gt = np.array(gt)
    preds = np.array(preds)
    return np.sqrt(np.sum((preds-gt)**2))

ALL_METRICS = [
    Metric("D_LAB",rmse,"lower")
]
