import json
import time

import pandas as pd
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="NIDS Dashboard", layout="wide")
st.title("Network Intrusion Detection Dashboard")

# ---- Sidebar controls ----
st.sidebar.header("Controls")
auto = st.sidebar.checkbox("Auto-refresh (every 3 s)", value=True)
min_risk = st.sidebar.slider("Minimum risk score", 0, 100, 0)
limit = st.sidebar.slider("Alerts to load", 10, 500, 100)

# ---- Fetch data from the API ----
try:
    stats = requests.get(f"{API}/stats", timeout=5).json()
    alerts = requests.get(
        f"{API}/alerts", params={"limit": limit, "min_risk": min_risk}, timeout=5
    ).json()
except requests.exceptions.RequestException:
    st.error("Cannot reach the API. Start uvicorn first (uvicorn main:app --reload in the api folder).")
    st.stop()

df = pd.DataFrame(alerts)

# ---- Summary cards ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total predictions", stats["total"])
c2.metric("Average risk", stats["avg_risk"])
c3.metric("Anomalies flagged", stats["anomalies"])
if not df.empty and df["actual_class"].notna().any():
    known = df.dropna(subset=["actual_class"])
    acc = (known["predicted_class"] == known["actual_class"]).mean()
    c4.metric("Accuracy (loaded rows)", f"{acc:.0%}")
else:
    c4.metric("Accuracy (loaded rows)", "n/a")

st.divider()

# ---- Charts ----
left, right = st.columns(2)

with left:
    st.subheader("Detections by attack type")
    if stats["by_class"]:
        st.bar_chart(pd.Series(stats["by_class"], name="count"))
    else:
        st.info("No predictions yet. Run the simulator.")

with right:
    st.subheader("Risk score over recent alerts")
    if not df.empty:
        st.line_chart(df.sort_values("id").set_index("id")["risk_score"])

st.divider()

# ---- Alert table ----
st.subheader("Alerts (highest risk first)")
if not df.empty:
    table = df[
        ["timestamp", "predicted_class", "actual_class", "confidence",
         "is_anomaly", "risk_score"]
    ].sort_values("risk_score", ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info("No alerts match the current filter.")
    # ---- SHAP explanation panel ----
st.divider()
st.subheader("Why was this flagged? (SHAP explanation)")

if not df.empty and "top_features" in df.columns:
    explainable = df[df["top_features"].notna() & (df["top_features"] != "{}")]
    if explainable.empty:
        st.info("No explanations yet. Run the simulator again.")
    else:
        explainable = explainable.sort_values("risk_score", ascending=False)
        labels = {
            r["id"]: f"#{r['id']} | predicted {r['predicted_class']} | risk {r['risk_score']}"
            for _, r in explainable.iterrows()
        }
        choice = st.selectbox("Pick an alert", list(labels.keys()), format_func=lambda i: labels[i])
        row = explainable[explainable["id"] == choice].iloc[0]
        feats = pd.Series(json.loads(row["top_features"]), name="SHAP value")

        c_a, c_b = st.columns(2)
        c_a.write(f"**Predicted:** {row['predicted_class']}  \n**Actual:** {row['actual_class']}")
        c_b.write("Positive bars pushed the model toward its predicted class. Negative bars pushed against it.")
        st.bar_chart(feats)

# ---- Auto-refresh ----
if auto:
    time.sleep(3)
    st.rerun()