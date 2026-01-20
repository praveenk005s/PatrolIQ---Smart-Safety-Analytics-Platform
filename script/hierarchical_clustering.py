import pandas as pd
import mlflow
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_hierarchical_sampled.csv"

# ==================================================
# Load Data
# ==================================================
print("📥 Loading feature-engineered data...")
df = pd.read_csv(INPUT_FILE)

FEATURES = ["Latitude", "Longitude", "Crime_Severity_Score"]

# ==================================================
# 🔥 CRITICAL: SAMPLE DATA (Hierarchical is O(n²))
# ==================================================
SAMPLE_SIZE = 3000   # SAFE LIMIT
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
X = df_sample[FEATURES]

# ==================================================
# Scaling
# ==================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================================
# Hierarchical Clustering Model
# ==================================================
hierarchical = AgglomerativeClustering(
    n_clusters=6,
    linkage="ward"
)

# ==================================================
# ✅ MLflow Experiment (FIXES YOUR ERROR)
# ==================================================
mlflow.set_experiment("PatrolIQ_Geo_Hierarchical")

with mlflow.start_run(run_name="Geo_Hierarchical_Sampled"):

    print("🚀 Running hierarchical clustering on sample...")

    labels = hierarchical.fit_predict(X_scaled)
    df_sample.loc[:, "Geo_Cluster_Hierarchical"] = labels

    sil_score = silhouette_score(X_scaled, labels)

    # 🔹 MLflow logging
    mlflow.log_param("algorithm", "AgglomerativeClustering")
    mlflow.log_param("linkage", "ward")
    mlflow.log_param("n_clusters", 6)
    mlflow.log_param("sample_size", SAMPLE_SIZE)
    mlflow.log_metric("silhouette_score", sil_score)

    print(f"✅ Silhouette Score (sample): {sil_score:.4f}")

# ==================================================
# Save Output
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_sample.to_csv(OUTPUT_FILE, index=False)

print("💾 Hierarchical clustering (sampled) saved")
print(f"📁 {OUTPUT_FILE}")
