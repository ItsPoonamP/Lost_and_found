"""
database.py — SQLite storage for every observation.

Schema:
    observations(id, track_id, class_name, timestamp,
                 frame_path, faiss_id, x1, y1, x2, y2, confidence)

Each row = one "sighting" of a tracked object at a moment in time.
"""

import sqlite3
from datetime import datetime
import config


class Database:
    def __init__(self):
        self._init_schema()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _connect(self):
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id    INTEGER NOT NULL,
                    class_name  TEXT    NOT NULL,
                    timestamp   TEXT    NOT NULL,
                    frame_path  TEXT    NOT NULL,
                    faiss_id    INTEGER NOT NULL,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    confidence  REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_track  ON observations(track_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_class  ON observations(class_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faiss  ON observations(faiss_id)")
            conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────────

    def insert(self, track_id, class_name, frame_path,
               faiss_id, bbox, confidence) -> int:
        """Insert observation. Returns the new row id."""
        x1, y1, x2, y2 = bbox
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO observations
                    (track_id, class_name, timestamp, frame_path,
                     faiss_id, x1, y1, x2, y2, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (track_id, class_name, ts, frame_path,
                  faiss_id, x1, y1, x2, y2, confidence))
            conn.commit()
            return cur.lastrowid

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        """Fetch observations by primary key list."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM observations WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, class_name: str = None, limit: int = 30) -> list[dict]:
        """Most recent observations, optionally filtered by class."""
        with self._connect() as conn:
            if class_name and class_name != "All":
                rows = conn.execute(
                    "SELECT * FROM observations WHERE class_name=? ORDER BY timestamp DESC LIMIT ?",
                    (class_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM observations ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Summary stats for the sidebar."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            rows  = conn.execute(
                "SELECT class_name, COUNT(*) AS cnt FROM observations "
                "GROUP BY class_name ORDER BY cnt DESC"
            ).fetchall()
        return {
            "total":   total,
            "classes": [{"name": r["class_name"], "count": r["cnt"]} for r in rows],
        }

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
