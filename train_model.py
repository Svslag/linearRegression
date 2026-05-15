from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "music_listeners.csv"
MODEL_PATH = BASE_DIR / "music_cluster_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"
OUTPUT_PATH = BASE_DIR / "clustered_music_listeners.csv"

FEATURES = ["daily_minutes", "rock_percent", "pop_percent", "hiphop_percent", "playlist_count"]

df = pd.read_csv(DATASET_PATH)
X = df[FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init="auto")
model.fit(X_scaled)

df["Cluster"] = model.labels_

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
df.to_csv(OUTPUT_PATH, index=False)

print("Unsupervised model trained successfully!")
print("Clusters created successfully!")
print("Saved file: clustered_music_listeners.csv")
