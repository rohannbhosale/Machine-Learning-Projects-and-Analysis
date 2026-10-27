import json
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "alerts.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                predicted_class TEXT,
                confidence REAL,
                is_anomaly INTEGER,
                anomaly_score REAL,
                risk_score REAL,
                actual_class TEXT
            )
        """)
        # Add the new column to the existing table (keeps your old rows)
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN top_features TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


def log_prediction(result, actual_class=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, predicted_class, confidence, is_anomaly, "
            "anomaly_score, risk_score, actual_class, top_features) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                result["predicted_class"],
                result["confidence"],
                int(result["is_anomaly"]),
                result["anomaly_score"],
                result["risk_score"],
                actual_class,
                json.dumps(result.get("top_features", {})),
            ),
        )

def recent_alerts(limit=50, min_risk=0):
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions WHERE risk_score >= ? ORDER BY id DESC LIMIT ?",
            (min_risk, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def stats():
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        by_class = conn.execute(
            "SELECT predicted_class, COUNT(*) AS n FROM predictions GROUP BY predicted_class"
        ).fetchall()
        avg_risk = conn.execute("SELECT AVG(risk_score) AS r FROM predictions").fetchone()["r"]
        anomalies = conn.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE is_anomaly = 1"
        ).fetchone()["n"]
    return {
        "total": total,
        "by_class": {r["predicted_class"]: r["n"] for r in by_class},
        "avg_risk": round(avg_risk, 1) if avg_risk is not None else 0,
        "anomalies": anomalies,
    }