import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

DROP_COLS = [
    "id",
    "is_ftp_login",
    "dloss",
    "sloss",
    "dwin",
    "ct_srv_src",
    "ct_dst_src_ltm",
    "tcprtt",
]

CATEGORICAL_COLS = ["proto", "service", "state"]

TARGET_BINARY = "label"
TARGET_MULTICLASS = "attack_cat"


def fit_preprocessor(train_df):
    df = train_df.copy()

    drop_now = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_now)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = [c for c in df.columns if c not in [TARGET_BINARY, TARGET_MULTICLASS]]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    preprocessor = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "drop_cols": drop_now,
    }

    return df, preprocessor


def transform(df, preprocessor):
    df = df.copy()

    drop_now = [c for c in preprocessor["drop_cols"] if c in df.columns]
    df = df.drop(columns=drop_now)

    for col in CATEGORICAL_COLS:
        le = preprocessor["encoders"][col]
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")
        if "unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown")
        df[col] = le.transform(df[col])

    feature_cols = preprocessor["feature_cols"]
    df[feature_cols] = preprocessor["scaler"].transform(df[feature_cols])

    return df


def save_preprocessor(preprocessor, path="../models/preprocessor.pkl"):
    joblib.dump(preprocessor, path)


def load_preprocessor(path="../models/preprocessor.pkl"):
    return joblib.load(path)