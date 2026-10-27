import pandas as pd
import numpy as np
import joblib
import shap
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR = SRC_DIR.parent / "models"
sys.path.append(str(SRC_DIR))

from preprocessing import transform, load_preprocessor
from risk_scoring import compute_risk_score

preprocessor = load_preprocessor(MODELS_DIR / "preprocessor.pkl")
supervised_model = joblib.load(MODELS_DIR / "supervised_model.pkl")
anomaly_model = joblib.load(MODELS_DIR / "anomaly_model.pkl")
target_encoder = joblib.load(MODELS_DIR / "target_encoder.pkl")
explainer = shap.TreeExplainer(supervised_model)

FEATURE_COLS = preprocessor["feature_cols"]


def predict_one(raw_row: dict, top_k: int = 5):
    df = pd.DataFrame([raw_row])
    processed = transform(df, preprocessor)
    X = processed[FEATURE_COLS]

    probs = supervised_model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    predicted_class = target_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    is_anomaly = anomaly_model.predict(X)[0] == -1
    anomaly_score = float(anomaly_model.decision_function(X)[0])

    risk = compute_risk_score(predicted_class, confidence, bool(is_anomaly), anomaly_score)

    # SHAP: which features pushed the model toward its predicted class
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        row_shap = shap_vals[pred_idx][0]
    else:
        row_shap = shap_vals[0, :, pred_idx]
    top = pd.Series(row_shap, index=FEATURE_COLS).sort_values(key=abs, ascending=False).head(top_k)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(anomaly_score, 3),
        "risk_score": risk,
        "top_features": {k: round(float(v), 3) for k, v in top.items()},
    }


if __name__ == "__main__":
    print("All models loaded.")
    raw_test = pd.read_csv(SRC_DIR.parent / "data" / "raw" / "UNSW_NB15_testing-set.csv")
    row = raw_test.iloc[0].drop(["attack_cat", "label"]).to_dict()
    print("Actual:", raw_test.iloc[0]["attack_cat"], "->", predict_one(row))