import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from preprocessing import fit_preprocessor, transform, save_preprocessor

train = pd.read_csv("../data/raw/UNSW_NB15_training-set.csv")
test = pd.read_csv("../data/raw/UNSW_NB15_testing-set.csv")

train_processed, preprocessor = fit_preprocessor(train)
test_processed = transform(test, preprocessor)

save_preprocessor(preprocessor, "../models/preprocessor.pkl")

train_processed.to_csv("../data/processed/train_processed.csv", index=False)
test_processed.to_csv("../data/processed/test_processed.csv", index=False)

print("Train processed shape:", train_processed.shape)
print("Test processed shape:", test_processed.shape)
print("\nSample of processed train data:")
print(train_processed.head())