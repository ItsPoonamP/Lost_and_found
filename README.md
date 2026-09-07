# Lost & Found — AI Object Retrieval System

An end-to-end computer vision pipeline that detects, tracks, and retrieves everyday objects from live camera footage.

---

## Architecture

```
Laptop Camera
      │
      ▼
 YOLOv11s          ← detects objects (backpack, phone, bottle…)
      │
      ▼
 BoT-SORT          ← assigns persistent track IDs across frames
      │
      ▼
 CLIP ViT-B-16     ← extracts 512-dim visual embedding from each crop
      │
   ┌──┴──┐
   │     │
SQLite  FAISS      ← stores (timestamp, frame_path) and embedding vectors
   └──┬──┘
      │
      ▼
 Streamlit UI      ← user uploads photo → cosine search → last seen location
```

---

## Paper Justification

| Component | Paper | Why |
|-----------|-------|-----|
| Detection | YOLOv11 (Ultralytics) | State-of-art real-time detector |
| Tracking  | BoT-SORT | Motion + appearance, better occlusion handling than ByteTrack |
| Embedding | CLIP (Radford et al., 2021) | Zero-shot visual similarity, no training needed |
| Search    | FAISS (Johnson et al., 2019) | Fast exact cosine search on CPU |

**Research gap addressed:** Existing trackers lose IDs after occlusion. This system is robust because CLIP embeddings allow retrieval even after tracker failure — the object's visual identity is stored permanently regardless of track ID changes.

---

## Setup

### 1. Clone / download this project

```bash
cd lost_found
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

First run will auto-download:
- `yolo11s.pt` (~22 MB)
- CLIP `ViT-B-16` weights (~340 MB)
- BoT-SORT tracker config (bundled with Ultralytics)

---

## Running

### Terminal 1 — Start the camera pipeline

```bash
python pipeline.py
```

A window opens showing the live camera feed with bounding boxes.

**Controls:**
- `q` → quit (saves index automatically)
- `s` → save FAISS index now
- `p` → pause / resume

Walk objects in front of the camera. The system saves observations to `data/`.

### Terminal 2 — Start the query UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

1. Go to **Find Object** tab
2. Upload a photo of your lost item
3. Click **Search**
4. See when and where it was last spotted

---

## Configuration (`config.py`)

| Setting | Default | Change to |
|---------|---------|-----------|
| `YOLO_MODEL` | `yolo11s.pt` | `yolo11n.pt` for faster CPU |
| `CLIP_MODEL` | `ViT-B-16` | `ViT-L-14` for better accuracy (needs GPU) |
| `EMBED_EVERY_N_FRAMES` | `15` | Lower = more observations, slower |
| `CONF_THRESHOLD` | `0.40` | Raise to reduce false detections |
| `TOP_K` | `5` | More results per search |
| `CAMERA_INDEX` | `0` | Change if using external webcam |

---

## Project Structure

```
lost_found/
├── pipeline.py          # Camera processing loop
├── app.py               # Streamlit query UI
├── config.py            # All settings
├── requirements.txt
│
├── core/
│   ├── detector.py      # YOLOv11s + BoT-SORT
│   ├── embedder.py      # CLIP encoding
│   ├── database.py      # SQLite read/write
│   └── search.py        # FAISS vector index
│
├── utils/
│   └── frame_utils.py   # Frame saving, drawing, HUD
│
└── data/                # Created automatically
    ├── frames/          # Saved JPEG snapshots
    ├── lost_found.db    # SQLite database
    ├── faiss.index      # FAISS binary index
    └── faiss_map.json   # FAISS position → DB row ID
```

---

## How Retrieval Works (for your report)

1. Every tracked crop is encoded by CLIP into a 512-dimensional embedding
2. Embeddings are L2-normalized → cosine similarity = inner product
3. FAISS `IndexFlatIP` stores all embeddings and performs exact nearest-neighbor search
4. At query time, the user's photo is encoded with the same CLIP model
5. FAISS returns the top-K most similar embeddings
6. Their timestamps and frame paths are fetched from SQLite
7. The object's **last known location and time** is displayed

---

## Troubleshooting

**Camera not opening**
```
cv2.error: Can't open camera 0
```
Try `CAMERA_INDEX = 1` in `config.py`

**BoT-SORT ReID model download slow**
Edit `config.py` and create a custom `botsort_no_reid.yaml`:
```yaml
tracker_type: botsort
with_reid: False
```
Then set `TRACKER = "botsort_no_reid.yaml"` — faster but less accurate re-ID.

**FAISS index empty after restart**
Pipeline saves index on quit (`q`). Always press `q` to exit, not Ctrl+C.
Or press `s` periodically to save manually.

**Out of memory**
Switch to `YOLO_MODEL = "yolo11n.pt"` and `CLIP_MODEL = "ViT-B-32"` in `config.py`.
