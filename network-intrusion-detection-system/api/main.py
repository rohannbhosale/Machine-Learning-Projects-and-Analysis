import sys
import json
import random
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from inference import predict_one
from database import init_db, log_prediction, recent_alerts, stats
app = FastAPI(title="NIDS API", version="0.1")
init_db()

# Load the raw test set once, so /sample can hand out realistic traffic rows
RAW_TEST = pd.read_csv(ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv")


class TrafficRecord(BaseModel):
    features: dict
    actual_class: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sample")
def sample():
    """Returns one random raw test row (plus its true label) for testing."""
    row = RAW_TEST.iloc[random.randrange(len(RAW_TEST))]
    record = json.loads(row.to_json())
    actual = record.pop("attack_cat")
    record.pop("label")
    return {"features": record, "actual_class": actual}


@app.post("/predict")
def predict(record: TrafficRecord):
    try:
        result = predict_one(record.features)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    log_prediction(result, record.actual_class)
    return result


@app.get("/alerts")
def alerts(limit: int = 50, min_risk: float = 0):
    return recent_alerts(limit, min_risk)


@app.get("/stats")
def get_stats():
    return stats()