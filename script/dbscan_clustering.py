import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_dbscan_sampled.csv"

# ==================================================
# Load Data
# ==================================================
print("📥 Loading feature-engineered data...")
df = pd.read_csv(INPUT_FILE)

FEATURES = ["Latitude", "Longitude", "Crime_Severity_Score"]

# ==================================================
# 🔥 SAMPLE DATA (DBSCAN is expensive)
# ==================================================
SAMPLE_SIZE = 20_000
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
X = df_sample[FEATURES]

# ==================================================
# Scaling
# ==================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================================
# DBSCAN Model
# ==================================================
dbscan = DBSCAN(
    eps=0.35,
    min_samples=50,
    metric="euclidean",
    n_jobs=1      # REQUIRED on Windows
)

# ==================================================
# ✅ FIX: SET EXPERIMENT (THIS SOLVES YOUR ERROR)
# ==================================================
mlflow.set_experiment("PatrolIQ_Geo_DBSCAN")

# ==================================================
# MLflow Tracking
# ==================================================
with mlflow.start_run(run_name="Geo_DBSCAN_Sampled"):

    print("🚀 Running DBSCAN on sampled data...")
    labels = dbscan.fit_predict(X_scaled)

    df_sample.loc[:, "Geo_Cluster_DBSCAN"] = labels

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).mean()

    if n_clusters > 1:
        mask = labels != -1
        sil_score = silhouette_score(X_scaled[mask], labels[mask])
    else:
        sil_score = -1.0

    # --------------------------------------------------
    # Log to MLflow
    # --------------------------------------------------
    mlflow.log_param("algorithm", "DBSCAN")
    mlflow.log_param("eps", 0.35)
    mlflow.log_param("min_samples", 50)
    mlflow.log_param("sample_size", SAMPLE_SIZE)

    mlflow.log_metric("n_clusters", n_clusters)
    mlflow.log_metric("noise_ratio", noise_ratio)
    mlflow.log_metric("silhouette_score", sil_score)

    print(f"🔢 Clusters found: {n_clusters}")
    print(f"⚠️ Noise ratio: {noise_ratio:.2%}")
    print(f"✅ Silhouette (sample): {sil_score:.4f}")

# ==================================================
# Save Output
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_sample.to_csv(OUTPUT_FILE, index=False)

print("💾 DBSCAN (sampled) results saved")
print(f"📁 {OUTPUT_FILE}")
