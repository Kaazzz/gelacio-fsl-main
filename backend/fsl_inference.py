"""
FSL inference engine.

- Singleton model loading (loaded once at startup via lifespan).
- Auto-detects d_model, num_layers, input_dim from checkpoint weights.
- Reads nhead, dropout, features from config.json in the model directory.
- Applies anchor-scale normalisation and optional velocity concatenation
  to match the training preprocessing pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config
from fsl_model import LandmarkTransformerV4
from sign_classes import get_class_names

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Globals (populated once at startup)
# ──────────────────────────────────────────────────────────────────────────────
_model: Optional[LandmarkTransformerV4] = None
_class_names: list[str] = []
_use_norm: bool = True
_use_vel: bool = False
_input_dim: int = 0          # raw feature dim fed to the model (after vel concat)
_base_dim: int = 0           # raw landmark feature dim before vel concat


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_model() -> None:
    """Load model from checkpoint. Called once during app lifespan startup."""
    global _model, _class_names, _use_norm, _use_vel, _input_dim, _base_dim

    # ── 1) Locate checkpoint ──────────────────────────────────────────────────
    ckpt_path = config.MODEL_PATH
    if not ckpt_path.exists():
        alt = config.MODEL_DIR / "best_by_acc.pt"
        if alt.exists():
            ckpt_path = alt
        else:
            logger.warning(
                "No checkpoint found at %s or %s. Running in mock mode.",
                config.MODEL_PATH, alt,
            )
            _class_names = get_class_names(config.NUM_CLASSES)
            return

    # ── 2) Load raw checkpoint ────────────────────────────────────────────────
    device = torch.device(config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    logger.info("Loaded checkpoint: %s", ckpt_path.name)

    # ── 3) Auto-detect hyperparameters from weight shapes ─────────────────────
    assert "inp.0.weight" in state, (
        "Cannot find 'inp.0.weight' in checkpoint — architecture mismatch?"
    )
    d_model = int(state["inp.0.weight"].shape[0])
    input_dim_ckpt = int(state["inp.0.weight"].shape[1])

    layer_ids: set[int] = set()
    for k in state.keys():
        if k.startswith("encoder.layers."):
            try:
                layer_ids.add(int(k.split(".")[2]))
            except ValueError:
                pass
    num_layers = (max(layer_ids) + 1) if layer_ids else 1

    logger.info(
        "Auto-detected: d_model=%d | num_layers=%d | input_dim=%d",
        d_model, num_layers, input_dim_ckpt,
    )

    # ── 4) Read training config for nhead / dropout / features ───────────────
    cfg: dict = {}
    if config.MODEL_CONFIG_PATH.exists():
        with open(config.MODEL_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        logger.info("Loaded model config: %s", config.MODEL_CONFIG_PATH.name)
    else:
        logger.info("No config.json found — using defaults (nhead=8, dropout=0.1)")

    # config.json stores model params under a nested "model" key
    model_cfg = cfg.get("model", {})
    nhead   = int(cfg.get("nhead",   model_cfg.get("nhead",   config.NHEAD)))
    dropout = float(cfg.get("dropout", model_cfg.get("dropout", config.DROPOUT)))

    features = cfg.get("features", [])
    use_norm = "anchor_scale_norm" in features
    use_vel  = "velocity_concat"   in features

    # If config had no features list, infer vel from input_dim parity.
    # MediaPipe Holistic JS produces 543 landmarks × 2 coords = 1086 floats max;
    # but the training base dim is whatever was used.  We detect via doubling:
    if not features:
        # We can't easily infer base_dim at load time without sample data.
        # Use a heuristic: if input_dim_ckpt is even, assume vel if > base guess.
        # Default safe: use_norm=True, infer vel from parity with input_dim_ckpt.
        use_norm = True
        # Guess base_dim from the mediapipe holistic landmark count.
        # pose(33) + left_hand(21) + right_hand(21) = 75 landmarks × 2 = 150
        # or × 3 = 225, or with face = 543 × 2 = 1086 etc.
        # We'll determine at runtime from the actual buffer sent by the client.
        use_vel = False   # conservative default; client must match training

    # ── 5) Determine num_classes ──────────────────────────────────────────────
    # Last layer of head is Linear(d_model → num_classes)
    head_weight_key = "head.2.weight"
    if head_weight_key in state:
        num_classes = int(state[head_weight_key].shape[0])
    else:
        num_classes = config.NUM_CLASSES
    logger.info("num_classes=%d", num_classes)

    # ── 6) Build and load model ───────────────────────────────────────────────
    model = LandmarkTransformerV4(
        input_dim=input_dim_ckpt,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    if isinstance(ckpt, dict):
        logger.info(
            "Checkpoint meta — epoch=%s val_loss=%s val_acc=%s",
            ckpt.get("epoch"), ckpt.get("val_loss"), ckpt.get("val_acc"),
        )

    # ── 7) Store globals ──────────────────────────────────────────────────────
    _model       = model
    _use_norm    = use_norm
    _use_vel     = use_vel
    _input_dim   = input_dim_ckpt
    _base_dim    = input_dim_ckpt // 2 if use_vel else input_dim_ckpt
    _class_names = get_class_names(num_classes)

    logger.info(
        "Model ready | use_norm=%s use_vel=%s input_dim=%d classes=%d device=%s",
        _use_norm, _use_vel, _input_dim, num_classes, config.DEVICE,
    )


def get_model() -> Optional[LandmarkTransformerV4]:
    return _model


def is_loaded() -> bool:
    return _model is not None


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _anchor_scale_norm(x: np.ndarray) -> np.ndarray:
    """
    Per-frame anchor-scale normalisation.
    x: (T, F) where F = L*C (L landmarks, C coords)
    """
    T, F = x.shape
    # Determine coordinate dimensionality
    if F % 3 == 0:
        C = 3
    elif F % 2 == 0:
        C = 2
    else:
        # Fallback: z-score normalise
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-6
        return (x - mu) / sd

    L = F // C
    X = x.reshape(T, L, C)                             # (T, L, C)
    anchor = X.mean(axis=1, keepdims=True)              # (T, 1, C)
    Xc = X - anchor                                     # (T, L, C)
    dist = np.sqrt((Xc ** 2).sum(axis=2))               # (T, L)
    scale = dist.mean(axis=1, keepdims=True)[..., None] + 1e-6  # (T, 1, 1)
    Xn = Xc / scale
    return Xn.reshape(T, F)


def preprocess(landmarks: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the same preprocessing as training.

    Args:
        landmarks: (T, F) float32 — raw landmark features
        mask:      (T,)   float32 — 1=valid, 0=padding

    Returns:
        x_proc:   (T, input_dim) float32 — preprocessed features
        mask_out: (T,) float32
    """
    x = landmarks.astype(np.float32)

    if _use_norm:
        x = _anchor_scale_norm(x)

    if _use_vel:
        v = np.zeros_like(x)
        v[1:] = x[1:] - x[:-1]
        x = np.concatenate([x, v], axis=1)              # (T, 2F)

    return x, mask.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(landmarks: list[list[float]], mask: list[float]) -> dict:
    """
    Run inference on a landmark buffer.

    Args:
        landmarks: list of T frames, each a flat list of F floats
        mask:      list of T floats (1=valid, 0=padding)

    Returns:
        dict with top_predictions, top_sign, top_probability, num_classes
    """
    if _model is None:
        # Mock mode — return uniform random predictions
        import random
        names = _class_names or [f"Sign_{i}" for i in range(config.NUM_CLASSES)]
        probs = sorted(
            [{"sign": n, "probability": round(random.random(), 3)} for n in names],
            key=lambda d: d["probability"],
            reverse=True,
        )[:5]
        return {
            "top_predictions": probs,
            "top_sign": probs[0]["sign"],
            "top_probability": probs[0]["probability"],
            "num_classes": len(names),
            "mock": True,
        }

    x_np = np.array(landmarks, dtype=np.float32)       # (T, F)
    m_np = np.array(mask, dtype=np.float32)            # (T,)

    x_np, m_np = preprocess(x_np, m_np)

    device = torch.device(config.DEVICE)
    x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)   # (1, T, F)
    m_t = torch.from_numpy(m_np).unsqueeze(0).to(device)   # (1, T)

    logits = _model(x_t, m_t)                               # (1, num_classes)
    probs  = torch.softmax(logits, dim=1)[0]                 # (num_classes,)

    k = min(5, probs.shape[0])
    top_probs, top_indices = torch.topk(probs, k)

    top_predictions = [
        {
            "sign": _class_names[idx.item()] if idx.item() < len(_class_names) else f"Sign_{idx.item()}",
            "probability": round(float(p.item()), 4),
        }
        for p, idx in zip(top_probs, top_indices)
    ]

    return {
        "top_predictions": top_predictions,
        "top_sign": top_predictions[0]["sign"],
        "top_probability": top_predictions[0]["probability"],
        "num_classes": probs.shape[0],
    }
