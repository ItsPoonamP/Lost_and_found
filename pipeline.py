"""
pipeline.py — Camera processing pipeline.

Run this in Terminal 1:
    python pipeline.py

Controls:
    q → quit and save index
    s → save FAISS index to disk now
    p → pause / resume

What it does every frame:
    1. Detect objects with YOLOv11s
    2. Track them with BoT-SORT (persistent IDs)
    3. Every EMBED_EVERY_N_FRAMES, extract CLIP embedding from crop
    4. Save frame + embedding to SQLite + FAISS
"""

import cv2
import sys
import config

from core.detector  import Detector
from core.embedder  import Embedder
from core.database  import Database
from core.search    import VectorSearch
from utils.frame_utils import save_frame, add_hud


class Pipeline:
    def __init__(self):
        print("\n── Lost & Found Pipeline ──────────────────────")
        print("Loading models…")
        self.detector  = Detector()
        print("  YOLOv11s + BoT-SORT ready")
        self.embedder  = Embedder()
        self.db        = Database()
        self.search    = VectorSearch()
        print("  SQLite + FAISS ready")
        print("───────────────────────────────────────────────\n")

        # frame counter per track_id for throttled embedding
        self._track_frames: dict[int, int] = {}
        self._db_count = 0
        self._paused   = False

    # ── Sampling logic ────────────────────────────────────────────────────────

    def _should_embed(self, track_id: int) -> bool:
        """Embed only every N frames to stay real-time on CPU."""
        count = self._track_frames.get(track_id, 0)
        self._track_frames[track_id] = count + 1
        return count % config.EMBED_EVERY_N_FRAMES == 0

    # ── Per-frame processing ──────────────────────────────────────────────────

    def _process(self, frame):
        detections, annotated = self.detector.detect_and_track(frame)

        for det in detections:
            tid = det["track_id"]
            if not self._should_embed(tid):
                continue

            crop = det["crop"]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # 1. CLIP embedding
            embedding = self.embedder.encode_image(crop)

            # 2. Save full frame snapshot
            frame_path = save_frame(frame, tid, det["class_name"])

            # 3. Add to FAISS with placeholder (-1) → get faiss_id
            faiss_id = self.search.add(embedding, obs_id=-1)

            # 4. Insert to DB → get real obs_id
            obs_id = self.db.insert(
                track_id   = tid,
                class_name = det["class_name"],
                frame_path = frame_path,
                faiss_id   = faiss_id,
                bbox       = det["bbox"],
                confidence = det["confidence"],
            )

            # 5. Update FAISS map with real obs_id
            self.search.update_obs_id(faiss_id, obs_id)
            self._db_count += 1

        return detections, annotated

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {config.CAMERA_INDEX}")
            sys.exit(1)

        print("Camera started.")
        print("Controls:  [q] quit   [s] save index   [p] pause\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed — retrying…")
                    continue

                if self._paused:
                    cv2.putText(frame, "PAUSED — press P to resume",
                                (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("Lost & Found | Pipeline", frame)
                else:
                    detections, annotated = self._process(frame)
                    display = add_hud(annotated, len(detections), self._db_count)
                    cv2.imshow("Lost & Found | Pipeline", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self.search.save()
                    print(f"[s] FAISS index saved ({self.search.total} vectors)")
                elif key == ord("p"):
                    self._paused = not self._paused
                    state = "paused" if self._paused else "resumed"
                    print(f"[p] Pipeline {state}")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        finally:
            self.search.save()
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nDone. {self._db_count} observations saved to database.")


if __name__ == "__main__":
    Pipeline().run()
