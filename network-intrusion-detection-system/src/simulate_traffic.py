import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000/predict"
DELAY_SECONDS = 1.0   # time between "packets"
N_ROWS = 200          # how many rows to send

raw_test = pd.read_csv(ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv")
sample = raw_test.sample(N_ROWS, random_state=None)

for i, (_, row) in enumerate(sample.iterrows(), start=1):
    record = json.loads(row.to_json())
    actual = record.pop("attack_cat")
    record.pop("label")

    resp = requests.post(API_URL, json={"features": record, "actual_class": actual})
    if resp.status_code != 200:
        print("Error:", resp.status_code, resp.text)
        break

    r = resp.json()
    print(f"[{i}/{N_ROWS}] actual={actual:<15} predicted={r['predicted_class']:<15} "
          f"risk={r['risk_score']:<5} anomaly={r['is_anomaly']}")
    time.sleep(DELAY_SECONDS)