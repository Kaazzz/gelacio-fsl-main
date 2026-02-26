"""
Load FSL class names from:
  Priority 1: backend/models/class_names.json  {"0": "Hello", "1": "Thank You", ...}
  Priority 2: backend/models/label_encoder.pkl (sklearn LabelEncoder)
  Fallback:   ["Sign_0", "Sign_1", ..., "Sign_N"]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def get_class_names(num_classes: int) -> list[str]:
    """Return a list of length num_classes with human-readable sign names."""

    # ── Priority 1: class_names.json ─────────────────────────────────────────
    json_path = config.CLASS_NAMES_PATH
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw: dict = json.load(f)
            # Support both {"0": "Hello"} and ["Hello", "Thank You"] formats
            if isinstance(raw, dict):
                names = [raw.get(str(i), f"Sign_{i}") for i in range(num_classes)]
            elif isinstance(raw, list):
                names = list(raw)
                # Pad if shorter
                while len(names) < num_classes:
                    names.append(f"Sign_{len(names)}")
            else:
                raise ValueError("Unexpected JSON format in class_names.json")
            logger.info("Loaded %d class names from class_names.json", len(names))
            return names
        except Exception as exc:
            logger.warning("Failed to load class_names.json: %s", exc)

    # ── Priority 2: label_encoder.pkl ────────────────────────────────────────
    pkl_path = config.MODEL_DIR / "label_encoder.pkl"
    if pkl_path.exists():
        try:
            import pickle
            with open(pkl_path, "rb") as f:
                le = pickle.load(f)
            # sklearn LabelEncoder stores classes_ as ndarray
            names = [str(c) for c in le.classes_]
            while len(names) < num_classes:
                names.append(f"Sign_{len(names)}")
            logger.info("Loaded %d class names from label_encoder.pkl", len(names))
            return names
        except Exception as exc:
            logger.warning("Failed to load label_encoder.pkl: %s", exc)

    # ── Fallback ──────────────────────────────────────────────────────────────
    logger.info("Using fallback class names: Sign_0 … Sign_%d", num_classes - 1)
    return [f"Sign_{i}" for i in range(num_classes)]
