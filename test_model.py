from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "music_cluster_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

FEATURES = ["daily_minutes", "rock_percent", "pop_percent", "hiphop_percent", "playlist_count"]

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Enter music listener information:")
row = {}

for feature in FEATURES:
    row[feature] = float(input(f"{feature}: "))

data = pd.DataFrame([row])
scaled_data = scaler.transform(data)
cluster = model.predict(scaled_data)[0]

print(f"Listener belongs to cluster: {cluster}")
