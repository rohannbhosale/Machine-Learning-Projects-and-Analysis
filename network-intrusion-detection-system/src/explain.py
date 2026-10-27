import pandas as pd
import numpy as np
import shap
import joblib

model = joblib.load("../models/supervised_model.pkl")
target_encoder = joblib.load("../models/target_encoder.pkl")
test = pd.read_csv("../data/processed/test_processed.csv")

feature_cols = [c for c in test.columns if c not in ["attack_cat", "label"]]
X_test = test[feature_cols]

explainer = shap.TreeExplainer(model)

print("Explainer created.")
print("Feature count:", len(feature_cols))

# Compute SHAP values for a small sample first (fast, avoids waiting on all 82k rows)
sample = X_test.iloc[:100]
shap_values = explainer.shap_values(sample)

print("SHAP values shape:", np.array(shap_values).shape)

# Pick one row to explain in detail
row_idx = 0
actual_class = target_encoder.inverse_transform([test["attack_cat"].iloc[row_idx]])[0] if isinstance(test["attack_cat"].iloc[row_idx], (int, np.integer)) else test["attack_cat"].iloc[row_idx]
predicted_class_idx = model.predict(sample.iloc[[row_idx]])[0]
predicted_class = target_encoder.inverse_transform([predicted_class_idx])[0]

print(f"\nRow {row_idx}")
print(f"Actual class: {actual_class}")
print(f"Predicted class: {predicted_class}")

# SHAP values for this row, for the predicted class
row_shap = shap_values[predicted_class_idx][row_idx] if isinstance(shap_values, list) else shap_values[row_idx, :, predicted_class_idx]

explanation = pd.Series(row_shap, index=feature_cols).sort_values(key=abs, ascending=False)
print("\nTop 5 features driving this prediction:")
print(explanation.head(5))

# ---- NEW: find and explain a CORRECT prediction too, for contrast ----
sample_actuals_encoded = target_encoder.transform(test["attack_cat"].iloc[:100])
sample_preds = model.predict(sample)

correct_indices = np.where(sample_preds == sample_actuals_encoded)[0]
print(f"\nCorrect predictions in this sample: {len(correct_indices)} / 100")

row_idx2 = correct_indices[0]
predicted_class_idx2 = sample_preds[row_idx2]
predicted_class2 = target_encoder.inverse_transform([predicted_class_idx2])[0]

print(f"\nRow {row_idx2} (correct prediction)")
print(f"Predicted class: {predicted_class2}")

row_shap2 = shap_values[predicted_class_idx2][row_idx2] if isinstance(shap_values, list) else shap_values[row_idx2, :, predicted_class_idx2]
explanation2 = pd.Series(row_shap2, index=feature_cols).sort_values(key=abs, ascending=False)
print("\nTop 5 features driving this correct prediction:")
print(explanation2.head(5))