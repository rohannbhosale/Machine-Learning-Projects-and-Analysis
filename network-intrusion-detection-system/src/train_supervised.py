import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import joblib

# Load processed data
train = pd.read_csv("../data/processed/train_processed.csv")
test = pd.read_csv("../data/processed/test_processed.csv")

print("Train:", train.shape)
print("Test:", test.shape)

from sklearn.preprocessing import LabelEncoder

# Encode the target labels (attack_cat) to numbers
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train["attack_cat"])
y_test = target_encoder.transform(test["attack_cat"])

# Features = everything except the two target columns
feature_cols = [c for c in train.columns if c not in ["attack_cat", "label"]]
X_train = train[feature_cols]
X_test = test[feature_cols]

print("Classes:", list(target_encoder.classes_))
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Softened class weights — sqrt scaling prevents tiny classes from dominating too aggressively
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
weights = np.sqrt(weights)  # soften extreme values
weights = np.clip(weights, a_min=None, a_max=5.0)  # cap max influence

class_weight_dict = dict(zip(classes, weights))
sample_weights = np.array([class_weight_dict[label] for label in y_train])

print("Softened class weights:")
for cls_idx, weight in class_weight_dict.items():
    print(f"  {target_encoder.classes_[cls_idx]}: {weight:.2f}")
    
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train, sample_weight=sample_weights)

print("Training complete.")

y_pred = model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

print("\n\n=== BASELINE: No class weighting (for comparison) ===")

baseline_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)

baseline_model.fit(X_train, y_train)  # no sample_weight this time

y_pred_baseline = baseline_model.predict(X_test)

print("=== Baseline Classification Report (no weighting) ===")
print(classification_report(y_test, y_pred_baseline, target_names=target_encoder.classes_))

# Save the chosen model (baseline, no class weighting) as the official supervised model
joblib.dump(baseline_model, "../models/supervised_model.pkl")
joblib.dump(target_encoder, "../models/target_encoder.pkl")

print("\nOfficial model saved: models/supervised_model.pkl")
print("Target encoder saved: models/target_encoder.pkl")