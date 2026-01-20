# ==================================================
# Imports
# ==================================================
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clustered.csv"

EXPORT_DIR = BASE_DIR / "streamlit_app/models"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Load Data
# ==================================================
df = pd.read_csv(INPUT_FILE)
FEATURES = ["Latitude", "Longitude", "Crime_Severity_Score"]
X = df[FEATURES]

# ==================================================
# Pipeline (Scaler + KMeans)
# ==================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", MiniBatchKMeans(
        n_clusters=6,
        batch_size=2048,
        random_state=42
    ))
])

# ==================================================
# MLflow
# ==================================================
mlflow.set_experiment("PatrolIQ_Geo_KMeans")

with mlflow.start_run(run_name="Geo_KMeans_Clustering"):
    pipeline.fit(X)
    labels = pipeline.predict(X)
    df["Geo_Cluster_KMeans"] = labels

    sample_idx = np.random.choice(len(X), 10_000, replace=False)
    X_sample = X.iloc[sample_idx]
    labels_sample = labels[sample_idx]

    X_sample_scaled = pipeline.named_steps["scaler"].transform(X_sample)
    sil_score = silhouette_score(X_sample_scaled, labels_sample)

    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name="PatrolIQ_Crime_Hotspot_Model"
    )

# ==================================================
# Save outputs
# ==================================================
df.to_csv(OUTPUT_FILE, index=False)

# ✅ SINGLE production artifact
joblib.dump(pipeline, EXPORT_DIR / "hotspot_model.pkl")

print("✅ Model exported for Streamlit Cloud")
