# Quick Setup Guide — GelacioFSL

**Get real-time sign language recognition running in under 10 minutes.**

---

## Prerequisites

| Software | Version | Check |
|----------|---------|-------|
| Python   | 3.9.x   | `python --version` |
| Node.js  | 18+     | `node --version` |
| npm      | 9+      | `npm --version` |
| Webcam   | any     | required for live inference |

---

## Step 1 — Place Model Files

Copy your trained checkpoint(s) into `backend/models/`:

```
backend/models/
├── best_by_loss.pt        ← primary (preferred)
├── best_by_acc.pt         ← fallback
├── config.json            ← training config (recommended)
└── class_names.json       ← optional class name map
```

**Minimum required:** at least one `.pt` file.

If no checkpoint is found, the backend starts in **mock mode** (random predictions) — useful for testing the UI without a model.

### config.json (recommended)

```json
{
  "nhead": 8,
  "dropout": 0.1,
  "features": ["anchor_scale_norm", "velocity_concat"]
}
```

The backend auto-detects `d_model`, `num_layers`, `input_dim`, and `num_classes` from the checkpoint weights. `config.json` only needs to supply `nhead`, `dropout`, and the `features` list.

### class_names.json (optional)

```json
{"0": "Hello", "1": "Thank You", "2": "Goodbye"}
```

Without this file, predictions display as `Sign_0`, `Sign_1`, etc.

---

## Step 2 — Backend Setup

```bash
cd backend

# Activate the existing virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install / update dependencies
pip install -r requirements.txt
```

**Dependencies installed:** `fastapi`, `uvicorn`, `torch`, `numpy`, `pydantic`, `aiofiles`

---

## Step 3 — Frontend Setup

```bash
cd frontend
npm install
```

This installs Next.js, MediaPipe packages (`@mediapipe/holistic`, `@mediapipe/drawing_utils`, `@mediapipe/camera_utils`), Axios, Tailwind CSS, and TypeScript.

---

## Step 4 — Start the Application

### Option A: start.bat (Windows — easiest)

Double-click `start.bat` in the project root. Two terminal windows open automatically — one for the backend, one for the frontend.

### Option B: Manual (cross-platform)

**Terminal 1 — Backend:**
```bash
cd backend
.venv\Scripts\activate
python main.py
```

Expected output:
```
INFO: Starting GelacioFSL API — loading model...
INFO: Auto-detected: d_model=256 | num_layers=3 | input_dim=150
INFO: Model ready | use_norm=True use_vel=False input_dim=150 classes=105
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Expected output:
```
▲ Next.js 14.0.3
  - Local: http://localhost:3000
✓ Ready in 2.5s
```

---

## Step 5 — Verify

- [ ] Open `http://localhost:8000/api/health` → should return `{"status": "healthy", "model_loaded": true, ...}`
- [ ] Open `http://localhost:3000` → dark neon UI loads
- [ ] Allow camera access when prompted
- [ ] Skeleton overlay appears on camera feed (cyan pose, green/pink hands)
- [ ] Perform a sign → predictions update within ~1 second
- [ ] Status bar shows FPS > 20 and buffer filling

---

## Common Issues

### Camera / skeleton not showing

- **Browser:** Use Chrome or Edge — best WebRTC + WebAssembly support
- **Permission:** Ensure camera access was granted (browser will prompt)
- **Console errors:** Press F12, check the Console tab for MediaPipe CDN errors
- **HTTPS:** MediaPipe requires a secure context — `localhost` is always allowed

### Predictions not updating

1. Check backend health: `http://localhost:8000/api/health`
2. Watch the **Buffer** indicator in the status bar — predictions only send when ≥ 10 valid frames are buffered
3. Check backend terminal for `Inference error` messages

### Dimension mismatch error on startup

The checkpoint `inp.0.weight` shape must match the feature vector your training pipeline produced. Common dimensions:

| Features | Landmarks | Coords | Dim |
|----------|-----------|--------|-----|
| pose+hands, xy only | 75 | 2 | 150 |
| pose+hands, xy + velocity | 75 | 2×2 | 300 |
| pose+hands, xyz | 75 | 3 | 225 |

Set `"features": ["velocity_concat"]` in `config.json` if your model was trained with velocity features.

### Backend starts but predictions are random (mock mode)

No checkpoint was found in `backend/models/`. Place `best_by_loss.pt` or `best_by_acc.pt` there and restart.

### Port already in use

```bash
# Windows — find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Frontend on a different port
npm run dev -- -p 3001
```

### npm install fails

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

Ensure Node.js ≥ 18: `node --version`

### Module not found (Python)

```bash
cd backend
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## What Each Component Does

| Component | Responsibility |
|-----------|---------------|
| `CameraView.tsx` | Opens webcam via `@mediapipe/camera_utils`, runs Holistic model, draws skeleton on `<canvas>`, pushes landmark frames to ring buffer |
| `LandmarkBuffer` | Ring buffer of 60 frames; returns padded `(60, F)` sequence + validity mask for the API |
| `StatusBar.tsx` | Shows live FPS, buffer fill %, and backend connection state |
| `PredictionDisplay.tsx` | Renders top-5 predictions as animated neon bars |
| `fsl_inference.py` | Loads checkpoint, auto-detects hyperparams, applies preprocessing, runs model |
| `fsl_model.py` | `LandmarkTransformerV4` definition (must exactly match training architecture) |
| `sign_classes.py` | Loads human-readable class names from JSON / pkl / fallback |

---

## Adjusting Inference Behaviour

| Setting | Location | Default |
|---------|----------|---------|
| Sliding window length | `frontend/app/components/CameraView.tsx` → `SEQ_LEN` | 60 |
| Minimum valid frames before sending | `CameraView.tsx` → `MIN_FRAMES` | 10 |
| Poll interval | `CameraView.tsx` → `POLL_INTERVAL_MS` | 500 ms |
| Backend port | `backend/config.py` → `PORT` | 8000 |
| Number of classes (fallback) | `backend/config.py` → `NUM_CLASSES` | 105 |

---

## Setup Complete

Once all checks pass:

- Skeleton overlays appear in real time at ~30 fps
- Predictions update every 500 ms with confidence bars
- Status bar shows FPS, buffer %, and connection state

For full documentation see [README.md](README.md).
