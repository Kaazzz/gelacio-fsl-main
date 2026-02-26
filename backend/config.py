import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_by_loss.pt"  # falls back to best_by_acc.pt in fsl_inference
MODEL_CONFIG_PATH = MODEL_DIR / "config.json"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# Create models directory if it doesn't exist
MODEL_DIR.mkdir(exist_ok=True)

# FSL Model configuration
NUM_CLASSES = 105
SEQ_LEN = 60       # number of frames in sliding window
MIN_FRAMES = 10    # minimum valid frames before sending to backend

# Feature extraction (overridden by config.json at runtime)
NHEAD = 8
DROPOUT = 0.1

# Server configuration
HOST = "0.0.0.0"
PORT = 8000

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
