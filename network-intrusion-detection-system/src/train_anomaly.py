import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

train = pd.read_csv("../data/processed/train_processed.csv")
test = pd.read_csv("../data/processed/test_processed.csv")

# Isolation Forest should learn ONLY what "normal" traffic looks like
train_normal = train[train["label"] == 0]

feature_cols = [c for c in train.columns if c not in ["attack_cat", "label"]]
X_train_normal = train_normal[feature_cols]

print("Normal-only training rows:", X_train_normal.shape)

anomaly_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # assume ~5% of "normal-looking" traffic might still be borderline/noisy
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)

anomaly_model.fit(X_train_normal)

print("Anomaly model training complete.")

X_test_all = test[feature_cols]
y_test_true_label = test["label"]  # 0 = normal, 1 = attack (ground truth)

# Isolation Forest returns 1 = normal, -1 = anomaly. Convert to match our label convention (0=normal, 1=attack)
raw_preds = anomaly_model.predict(X_test_all)
anomaly_preds = np.where(raw_preds == -1, 1, 0)

from sklearn.metrics import classification_report
print("=== Anomaly Detector Performance (flagging attacks as anomalies) ===")
print(classification_report(y_test_true_label, anomaly_preds, target_names=["Normal", "Attack"]))

anomaly_model_v2 = IsolationForest(
    n_estimators=200,
    contamination=0.2,   # more sensitive — expect ~20% of traffic to look unusual
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)

anomaly_model_v2.fit(X_train_normal)

raw_preds_v2 = anomaly_model_v2.predict(X_test_all)
anomaly_preds_v2 = np.where(raw_preds_v2 == -1, 1, 0)

print("=== Anomaly Detector v2 (contamination=0.2) ===")
print(classification_report(y_test_true_label, anomaly_preds_v2, target_names=["Normal", "Attack"]))

test_with_preds = test.copy()
test_with_preds["anomaly_pred"] = anomaly_preds_v2

# Only look at rows that are actually attacks, broken down by type
attack_rows = test_with_preds[test_with_preds["label"] == 1]
detection_rate_by_type = attack_rows.groupby("attack_cat")["anomaly_pred"].mean().sort_values(ascending=False)

print("=== Anomaly Detection Rate by Attack Type ===")
print(detection_rate_by_type)

joblib.dump(anomaly_model_v2, "../models/anomaly_model.pkl")
print("Anomaly model saved: models/anomaly_model.pkl")