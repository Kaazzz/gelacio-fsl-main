# GelacioFSL – Real-time Sign Language Recognition

**Real-time Filipino Sign Language (FSL) recognition powered by a Transformer-based deep learning model**

GelacioFSL is a local-deployment application that uses your webcam and MediaPipe Holistic to extract body/hand landmarks in the browser, then runs them through a trained **LandmarkTransformerV4** model on the backend to recognise FSL signs in real time.

## Features

- **Real-time inference** — MediaPipe runs at ~30 fps in the browser; predictions update every 500 ms
- **Skeleton overlay** — neon-coloured pose and hand landmarks drawn on a canvas overlay
- **Top-5 predictions** — animated confidence bars with glow effects
- **105 sign classes** — auto-loads class names from `class_names.json` or falls back to `Sign_N`
- **Auto-detects model hyperparameters** — `d_model`, `num_layers`, `input_dim` read from checkpoint weights
- **Dark neon UI** — `#0f0f1a` background, cyan / green / pink accents
- **Local deployment** — all processing on your machine; no data leaves your device

## Architecture

### System Overview

```
gelacio-fsl-main/
├── backend/                        # Python FastAPI backend (port 8000)
│   ├── main.py                    # API endpoints with lifespan startup
│   ├── fsl_model.py               # LandmarkTransformerV4 + AttentionPool
│   ├── fsl_inference.py           # Singleton loader, preprocessing, inference
│   ├── sign_classes.py            # Class name loader (JSON / pkl / fallback)
│   ├── config.py                  # Paths, SEQ_LEN, NUM_CLASSES, device
│   ├── requirements.txt           # Python dependencies
│   └── models/
│       ├── best_by_loss.pt        # Trained checkpoint (primary)
│       ├── best_by_acc.pt         # Trained checkpoint (fallback)
│       ├── config.json            # Training config (nhead, dropout, features)
│       └── class_names.json       # Optional: {"0": "Hello", "1": "Thank You", ...}
└── frontend/                       # Next.js 14 application (port 3000)
    └── app/
        ├── components/
        │   ├── HeroSection.tsx    # Dark neon branding header
        │   ├── CameraView.tsx     # WebRTC + MediaPipe + canvas skeleton overlay
        │   ├── PredictionDisplay.tsx  # Top-5 animated confidence bars
        │   └── StatusBar.tsx      # FPS / buffer fill / connection status
        ├── lib/
        │   ├── api.ts             # predictLandmarks() → POST /api/predict-landmarks
        │   ├── landmarkBuffer.ts  # Ring-buffer: push frames, get padded sequence + mask
        │   └── utils.ts           # formatPct, getNeonColor, normalizeBuffer
        ├── page.tsx               # Main layout (camera | predictions)
        └── globals.css            # Dark neon CSS variables + utilities
```

### Model Architecture: LandmarkTransformerV4

```
Input: (T, input_dim)   ← T frames × F landmark features
  │
  ├── inp: Linear(input_dim → d_model) + LayerNorm + Dropout
  │
  ├── encoder: TransformerEncoder
  │     └── N × TransformerEncoderLayer
  │           (norm_first=True, GELU, batch_first=True, nhead=8, dim_ff=4×d_model)
  │
  ├── pool: AttentionPool  ← learned score, masked softmax, weighted sum
  │
  └── head: LayerNorm + Dropout + Linear(d_model → num_classes)

Output: logits (num_classes,)
```

**Auto-detection at load time:**
- `d_model` ← `inp.0.weight.shape[0]`
- `input_dim` ← `inp.0.weight.shape[1]`
- `num_layers` ← max layer index in `encoder.layers.*` + 1
- `num_classes` ← `head.2.weight.shape[0]`
- `nhead`, `dropout`, feature flags ← `models/config.json` (with safe defaults)

### Data Flow

```
Browser (30 fps)
  MediaPipe Holistic
    └── pose(33) + left_hand(21) + right_hand(21) = 75 landmarks × 2 = 150 floats/frame
          │
          ├── Draw skeleton on <canvas> overlay (neon colours)
          │
          └── Push to LandmarkBuffer (ring buffer, capacity=SEQ_LEN=60)
                  │
                  every 500 ms (if ≥ MIN_FRAMES valid frames)
                  │
                  POST /api/predict-landmarks
                  { landmarks: (60, 150), mask: (60,) }
                          │
                    Backend preprocessing:
                      anchor-scale normalisation
                      optional velocity concat
                          │
                    LandmarkTransformerV4
                          │
                    top-5 predictions JSON
                          │
                  PredictionDisplay (animated bars)
```

## Prerequisites

- **Python 3.9.x** — backend runtime (existing venv at `backend/.venv`)
- **Node.js 18+** — frontend build system
- **Webcam** — required for real-time landmark extraction
- **Trained checkpoint** — `best_by_loss.pt` or `best_by_acc.pt` placed in `backend/models/`

## Model Files Setup

Model files are **not included** in the repository. Place them at:

```
backend/models/
├── best_by_loss.pt        ← primary checkpoint (preferred)
├── best_by_acc.pt         ← fallback checkpoint
├── config.json            ← training config (recommended)
└── class_names.json       ← optional class name map
```

**`config.json` format** (from training run):
```json
{
  "nhead": 8,
  "dropout": 0.1,
  "features": ["anchor_scale_norm", "velocity_concat"]
}
```

**`class_names.json` format** (optional):
```json
{"0": "Hello", "1": "Thank You", "2": "Goodbye", ...}
```

If `config.json` is absent, the backend defaults to `nhead=8`, `dropout=0.1`, `anchor_scale_norm=True`, `velocity_concat=False`.

If no checkpoint is found, the backend starts in **mock mode** and returns random predictions so the UI can still be tested.

## Installation

### Backend

```bash
cd backend
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `torch`, `numpy`, `pydantic`, `aiofiles`

### Frontend

```bash
cd frontend
npm install
```

Key new dependencies: `@mediapipe/holistic`, `@mediapipe/drawing_utils`, `@mediapipe/camera_utils`

## Running the Application

### Option 1: start.bat (Windows — recommended)

Double-click `start.bat` in the project root. This opens two terminals (backend + frontend) and launches both servers.

### Option 2: Manual

**Terminal 1 — Backend:**
```bash
cd backend
.venv\Scripts\activate
python main.py
```
Backend ready at `http://localhost:8000`

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```
Frontend ready at `http://localhost:3000`

## Usage

1. Open `http://localhost:3000` in Chrome or Edge (best MediaPipe support)
2. Allow camera access when prompted
3. The skeleton overlay appears immediately — pose in **cyan**, left hand in **green**, right hand in **pink**
4. Perform a sign — predictions update every 500 ms once the frame buffer has at least 10 valid frames
5. The top-5 panel shows animated confidence bars; the largest sign is displayed prominently

## API Reference

### `GET /api/health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 105,
  "seq_len": 60,
  "device": "cpu"
}
```

### `POST /api/predict-landmarks`

**Request:**
```json
{
  "landmarks": [[f0, f1, ...], ...],
  "mask": [1.0, 1.0, 0.0, ...]
}
```
- `landmarks`: `(T, F)` array — T=SEQ_LEN frames, F=feature dim
- `mask`: `(T,)` — `1.0` for valid frames, `0.0` for padding

**Response:**
```json
{
  "top_predictions": [
    {"sign": "Hello", "probability": 0.784},
    {"sign": "Thank You", "probability": 0.421}
  ],
  "top_sign": "Hello",
  "top_probability": 0.784,
  "num_classes": 105
}
```

Interactive docs: `http://localhost:8000/docs`

## Tech Stack

### Backend
- **FastAPI** — async web framework
- **PyTorch** — model inference
- **NumPy** — preprocessing (anchor-scale norm, velocity concat)
- **Pydantic v2** — request / response validation
- **Uvicorn** — ASGI server
- **Python 3.9.x**

### Frontend
- **Next.js 14** — React framework with App Router
- **@mediapipe/holistic** — landmark extraction (pose + hands)
- **@mediapipe/drawing_utils** — skeleton drawing utilities
- **@mediapipe/camera_utils** — webcam feed management
- **TypeScript 5** — type safety
- **Tailwind CSS 3** — utility-first styling
- **Axios** — HTTP client

## Troubleshooting

**Skeleton not appearing**
- Check browser console for MediaPipe CDN errors
- Ensure camera permission was granted
- Try Chrome or Edge (Firefox has limited WebRTC support)

**Predictions not updating**
- Verify backend is running: `http://localhost:8000/api/health`
- Check the buffer fill indicator in the status bar — it must reach the minimum before requests are sent
- Check backend terminal for inference errors

**Wrong number of classes / dimension mismatch**
- Verify `config.json` matches your training config
- Check backend startup logs for `d_model`, `num_layers`, `input_dim` values
- The feature vector dimension (150 for pose+hands ×2 coords, or 300 with velocity) must match the checkpoint's `inp.0.weight.shape[1]`

**Model not loading**
- Ensure `.pt` file is in `backend/models/`
- Backend will start in mock mode if no checkpoint is found (random predictions)

**Port conflicts**
```bash
# Windows — kill port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## License

Provided for educational and research purposes.
