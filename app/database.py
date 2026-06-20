import json
import sqlite3
from datetime import datetime

from app.config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                model_id TEXT NOT NULL,
                top1_class TEXT NOT NULL,
                top1_confidence REAL NOT NULL,
                top3_json TEXT NOT NULL,
                inference_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction(filename, model_id, top3, inference_ms):
    top1 = top3[0]
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (filename, model_id, top1_class, top1_confidence, top3_json, inference_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                model_id,
                top1["class"],
                top1["confidence"],
                json.dumps(top3, ensure_ascii=False),
                inference_ms,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def get_history(limit=50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_statistics():
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(top1_confidence) FROM predictions"
        ).fetchone()[0]
        avg_time = conn.execute(
            "SELECT AVG(inference_ms) FROM predictions"
        ).fetchone()[0]
        by_class = conn.execute(
            """
            SELECT top1_class, COUNT(*) as cnt
            FROM predictions GROUP BY top1_class ORDER BY cnt DESC LIMIT 5
            """
        ).fetchall()
    return {
        "total": total or 0,
        "avg_confidence": round(avg_conf or 0, 3),
        "avg_inference_ms": round(avg_time or 0, 1),
        "top_classes": [{"class": r[0], "count": r[1]} for r in by_class],
    }


def export_history_json():
    history = get_history(limit=1000)
    return json.dumps(history, ensure_ascii=False, indent=2)
