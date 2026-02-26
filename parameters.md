# ============================================================
# ONE CELL: Plot curves + AUTO-build model from checkpoint + final eval (val + test)
# - Auto-detects d_model + num_layers from checkpoint tensors
# - Rebuilds val split from CACHE_DIR/train_meta.json (SEED=42, VAL_RATIO=0.15)
# - Uses features from config.json if present; falls back to inferring from INPUT_DIM
# - Loads best_by_loss.pt else best_by_acc.pt
# - Prints: val/test top1+top5, confusion totals, per-class acc best/worst
# ============================================================

import os, json, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

assert "CACHE_DIR" in globals(), "Define CACHE_DIR before running."
assert "NUM_CLASSES" in globals(), "Define NUM_CLASSES before running."

RUN_DIR = "/content/drive/MyDrive/Gelacio/fsl/fsl/runs/20260219_175659_v4_more_realistic"
assert os.path.isdir(RUN_DIR), f"RUN_DIR not found: {RUN_DIR}"

# --------------------------
# 1) Plot curves from history.csv (if present)
# --------------------------
hist_path = os.path.join(RUN_DIR, "history.csv")
cfg_path  = os.path.join(RUN_DIR, "config.json")
config = {}
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        config = json.load(f)

if os.path.exists(hist_path):
    rows = []
    with open(hist_path, "r") as f:
        r = csv.reader(f)
        header = next(r, None)
        for line in r:
            if line:
                rows.append(line)
    arr = np.array(rows, dtype=np.float64)
    epoch = arr[:, 0]
    train_loss = arr[:, 1]
    train_acc  = arr[:, 2]
    val_loss   = arr[:, 3]
    val_acc    = arr[:, 4]
    lr         = arr[:, 5]

    plt.figure()
    plt.plot(epoch, train_loss, label="train_loss")
    plt.plot(epoch, val_loss,   label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curves"); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epoch, train_acc, label="train_acc")
    plt.plot(epoch, val_acc,   label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy curves"); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epoch, lr, label="lr")
    plt.xlabel("epoch"); plt.ylabel("lr"); plt.title("Learning rate"); plt.legend()
    plt.show()
else:
    print("No history.csv found; skipping plots:", hist_path)

# --------------------------
# 2) Load checkpoint (prefer best_by_loss)
# --------------------------
ckpt_loss = os.path.join(RUN_DIR, "best_by_loss.pt")
ckpt_acc  = os.path.join(RUN_DIR, "best_by_acc.pt")
ckpt_path = ckpt_loss if os.path.exists(ckpt_loss) else ckpt_acc
assert os.path.exists(ckpt_path), f"No checkpoint found in RUN_DIR: {RUN_DIR}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

# --------------------------
# 3) Auto-detect model hyperparams from state_dict
# --------------------------
# d_model from inp.0.weight: [d_model, input_dim]
assert "inp.0.weight" in state, "Can't find inp.0.weight in checkpoint; different architecture?"
d_model = int(state["inp.0.weight"].shape[0])
input_dim_ckpt = int(state["inp.0.weight"].shape[1])

# num_layers by counting encoder.layers.{i}.*
layer_ids = set()
for k in state.keys():
    if k.startswith("encoder.layers."):
        try:
            layer_ids.add(int(k.split(".")[2]))
        except:
            pass
num_layers = (max(layer_ids) + 1) if layer_ids else 0

# infer nhead crudely: many people keep nhead=8, but you can also detect by in_proj_weight shape = [3*d_model, d_model]
# nhead doesn't affect weight shapes directly except inside MultiheadAttention, but in_proj shapes match regardless of nhead.
# We'll default to 8 unless config contains it.
nhead = int(config.get("nhead", 8)) if isinstance(config, dict) else 8
dropout = float(config.get("dropout", 0.1)) if isinstance(config, dict) else 0.1

print(f"Checkpoint: {os.path.basename(ckpt_path)} | d_model={d_model} | num_layers={num_layers} | input_dim={input_dim_ckpt}")

# --------------------------
# 4) Dataset (norm + vel) aligned with INPUT_DIM used in checkpoint
# --------------------------
train_meta_path = os.path.join(CACHE_DIR, "train_meta.json")
test_meta_path  = os.path.join(CACHE_DIR, "test_meta.json")
assert os.path.exists(train_meta_path), f"Missing {train_meta_path}"
assert os.path.exists(test_meta_path),  f"Missing {test_meta_path}"

with open(train_meta_path, "r") as f:
    train_meta_all = json.load(f)
with open(test_meta_path, "r") as f:
    test_meta = json.load(f)

def normalize_meta(meta):
    if isinstance(meta, dict):
        meta = list(meta.values())
    if isinstance(meta, list) and len(meta) and isinstance(meta[0], str):
        fixed = []
        for p in meta:
            base = os.path.basename(p)
            label = -1
            parts = base.split("_")
            if len(parts) >= 2:
                try: label = int(parts[1])
                except: label = -1
            fixed.append({"npz": p, "label": label})
        meta = fixed
    assert isinstance(meta, list) and len(meta), "Meta must be a non-empty list"
    assert isinstance(meta[0], dict) and "npz" in meta[0], "Meta items must have key 'npz'"
    for it in meta:
        if "label" not in it:
            it["label"] = -1
    return meta

train_meta_all = normalize_meta(train_meta_all)
test_meta      = normalize_meta(test_meta)

# Decide feature flags:
# - Prefer config["features"]
# - Else infer from input_dim_ckpt (318 = no vel, 636 = vel with base 318)
features = config.get("features", []) if isinstance(config, dict) else []
use_norm = ("anchor_scale_norm" in features)
use_vel  = ("velocity_concat" in features)

# If config didn't say, infer from input_dim_ckpt vs base feature dim in npz
# (npz x feature dim = baseF)
sample_npz = np.load(train_meta_all[0]["npz"])
baseF = int(sample_npz["x"].shape[1])
sample_npz.close()

if not features:
    use_vel = (input_dim_ckpt == 2 * baseF)
    use_norm = True  # you can change this if you want; usually keep True

# sanity: ensure our computed input dim matches checkpoint
expected_dim = baseF * (2 if use_vel else 1)
assert expected_dim == input_dim_ckpt, f"Feature dim mismatch: baseF={baseF}, use_vel={use_vel} => {expected_dim}, ckpt expects {input_dim_ckpt}"

SEED = 42
VAL_RATIO = 0.15
labels = [int(m.get("label", -1)) for m in train_meta_all]
use_strat = all(isinstance(l, int) and l >= 0 for l in labels)

train_meta, val_meta = train_test_split(
    train_meta_all,
    test_size=VAL_RATIO,
    random_state=SEED,
    shuffle=True,
    stratify=labels if use_strat else None
)

class LandmarkNPZDatasetEval(Dataset):
    def __init__(self, meta, use_norm=False, use_vel=False):
        self.meta = meta
        self.use_norm = use_norm
        self.use_vel = use_vel

    def __len__(self):
        return len(self.meta)

    def _norm_anchor_scale(self, x):
        T, F = x.shape
        C = 3 if (F % 3 == 0) else (2 if (F % 2 == 0) else None)
        if C is None:
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-6
            return (x - mu) / sd
        L = F // C
        X = x.reshape(T, L, C)
        anchor = X.mean(axis=1, keepdims=True)
        Xc = X - anchor
        dist = np.sqrt((Xc ** 2).sum(axis=2))
        scale = dist.mean(axis=1, keepdims=True)[..., None] + 1e-6
        Xn = Xc / scale
        return Xn.reshape(T, F)

    def __getitem__(self, idx):
        item = self.meta[idx]
        data = np.load(item["npz"])
        x = data["x"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        label = int(item.get("label", -1))
        if label == -1 and "label" in data:
            label = int(data["label"])
        data.close()

        if self.use_norm:
            x = self._norm_anchor_scale(x)
        if self.use_vel:
            v = np.zeros_like(x)
            v[1:] = x[1:] - x[:-1]
            x = np.concatenate([x, v], axis=1)

        return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(label, dtype=torch.long)

train_ds = LandmarkNPZDatasetEval(train_meta, use_norm=use_norm, use_vel=use_vel)
val_ds   = LandmarkNPZDatasetEval(val_meta,   use_norm=use_norm, use_vel=use_vel)
test_ds  = LandmarkNPZDatasetEval(test_meta,  use_norm=use_norm, use_vel=use_vel)

BATCH = int(config.get("batch", 32)) if isinstance(config, dict) else 32
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

print(f"Split sizes: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)} | input_dim={input_dim_ckpt} | norm={use_norm} vel={use_vel}")

# --------------------------
# 5) Model definition matching your checkpoint head style: LN -> Dropout -> Linear
# --------------------------
class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h, mask):
        logits = self.score(h.float()).squeeze(-1)
        logits = logits.masked_fill(mask <= 0, -1e4)
        w = torch.softmax(logits, dim=1).to(h.dtype)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        return pooled, w

class LandmarkTransformerV4(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask):
        h = self.inp(x)
        key_pad = (mask <= 0)
        h = self.encoder(h, src_key_padding_mask=key_pad)
        pooled, _ = self.pool(h, mask)
        return self.head(pooled)

model = LandmarkTransformerV4(
    input_dim=input_dim_ckpt,
    num_classes=NUM_CLASSES,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dropout=dropout
).to(device)

# Load weights (strict should now work)
model.load_state_dict(state, strict=True)
model.eval()

if isinstance(ckpt, dict):
    print(f"Loaded checkpoint ok. epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')} val_acc={ckpt.get('val_acc')}")
else:
    print("Loaded checkpoint ok.")

# --------------------------
# 6) Eval (top-1 + top-5) + per-class + confusion totals
# --------------------------
@torch.no_grad()
def eval_loader(loader):
    all_true, all_pred, all_top5 = [], [], []
    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            logits = model(x, mask)

        pred = logits.argmax(dim=1)
        top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices

        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_top5.append(top5.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_top5 = np.concatenate(all_top5)

    top1 = float((y_pred == y_true).mean())
    top5_acc = float(np.mean([y_true[i] in y_top5[i] for i in range(len(y_true))]))
    return y_true, y_pred, top1, top5_acc

def per_class_stats(y_true, y_pred, num_classes):
    out = []
    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            continue
        cor = int((y_pred[idx] == y_true[idx]).sum())
        tot = int(len(idx))
        out.append((c, cor / tot, cor, tot))
    return out

y_val, yv_pred, val_top1, val_top5 = eval_loader(val_loader)
y_test, yt_pred, test_top1, test_top5 = eval_loader(test_loader)

print(f"\nVAL  accuracy: {val_top1:.4f} | top-5: {val_top5:.4f}")
print(f"TEST accuracy: {test_top1:.4f} | top-5: {test_top5:.4f}")

cm = confusion_matrix(y_test, yt_pred, labels=list(range(NUM_CLASSES)))
correct = int(np.trace(cm))
total = int(cm.sum())
print(f"\nConfusion matrix totals (TEST): Total={total} | Correct={correct} | Wrong={total-correct}")

stats = per_class_stats(y_test, yt_pred, NUM_CLASSES)
accs = [s[1] for s in stats]
print(f"\nPer-class accuracy summary (TEST, only classes present):")
print(f"Classes in test: {len(stats)} / {NUM_CLASSES}")
print(f"Mean per-class acc: {float(np.mean(accs)):.4f}")
print(f"Median per-class acc: {float(np.median(accs)):.4f}")

stats_sorted = sorted(stats, key=lambda x: x[1])
print("\nWorst 10 classes (class_id: acc, correct/total):")
for c, a, cor, tot in stats_sorted[:10]:
    print(f"{c:>3}: {a:.4f} ({cor}/{tot})")

print("\nBest 10 classes (class_id: acc, correct/total):")
for c, a, cor, tot in stats_sorted[-10:][::-1]:
    print(f"{c:>3}: {a:.4f} ({cor}/{tot})")

# small confusion matrix preview (top-left 30x30)
cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-9)
plt.figure(figsize=(7,6))
plt.imshow(cm_norm[:30, :30], aspect="auto")
plt.title("Confusion Matrix (row-normalized) preview: classes 0-29")
plt.xlabel("pred"); plt.ylabel("true")
plt.colorbar()
plt.show()
