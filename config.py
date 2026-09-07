"""
config.py — All settings in one place.
Change CLIP_MODEL or YOLO_MODEL here to switch models easily.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
FRAMES_DIR  = os.path.join(DATA_DIR, "frames")
DB_PATH     = os.path.join(DATA_DIR, "lost_found.db")
FAISS_INDEX = os.path.join(DATA_DIR, "faiss.index")
FAISS_MAP   = os.path.join(DATA_DIR, "faiss_map.json")

os.makedirs(FRAMES_DIR, exist_ok=True)

# ── Detection ─────────────────────────────────────────────────────────────────
# yolo11n.pt = fastest (nano), yolo11s.pt = better accuracy (small)
YOLO_MODEL      = "yolo11s.pt"
CONF_THRESHOLD  = 0.40
IOU_THRESHOLD   = 0.50
TRACKER         = "botsort.yaml"   # BoT-SORT built into Ultralytics

# COCO class IDs for everyday "losable" objects
TRACKED_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    73: "book",
    76: "scissors",
    77: "teddy bear",
}

# ── Embedding (CLIP) ───────────────────────────────────────────────────────────
# ViT-B-16  → 512-dim, fast, good accuracy   ← recommended for laptop
# ViT-L-14  → 768-dim, slower, best accuracy ← if you have GPU
CLIP_MODEL      = "ViT-B-16"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM   = 512   # must match CLIP_MODEL output

# ── Sampling ──────────────────────────────────────────────────────────────────
# Save embedding every N frames per track (avoid saving every frame = too slow)
EMBED_EVERY_N_FRAMES = 15   # ~0.5s at 30fps

# ── Search ─────────────────────────────────────────────────────────────────────
TOP_K = 5   # number of results to return

# ── Camera ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
