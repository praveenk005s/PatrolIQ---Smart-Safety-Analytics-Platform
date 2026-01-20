import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_temporal_clustered.csv"

# ==================================================
# Load data
# ==================================================
print("📥 Loading feature-engineered dataset...")
df = pd.read_csv(INPUT_FILE)

TEMPORAL_FEATURES = [
    "Hour",
    "Day_of_Week",
    "Month",
    "Is_Weekend"
]

X = df[TEMPORAL_FEATURES]

# ==================================================
# MLflow Experiment Setup
# ==================================================
EXPERIMENT_NAME = "PatrolIQ_Temporal_Clustering"
mlflow.set_experiment(EXPERIMENT_NAME)

# ==================================================
# Pipeline (Scaler + KMeans)
# ==================================================
K = 4  # recommended 3–5 clusters

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", MiniBatchKMeans(
        n_clusters=K,
        batch_size=10000,
        random_state=42
    ))
])

# ==================================================
# MLflow Tracking
# ==================================================
with mlflow.start_run(run_name="Temporal_KMeans"):
    print("🚀 Running temporal clustering...")

    labels = pipeline.fit_predict(X)
    df["Temporal_Cluster"] = labels

    # 🔹 Silhouette on SAMPLE (memory safe)
    X_sample, labels_sample = resample(
        X,
        labels,
        n_samples=10000,
        random_state=42
    )

    X_sample_scaled = pipeline.named_steps["scaler"].transform(X_sample)
    sil_score = silhouette_score(X_sample_scaled, labels_sample)

    # 🔹 MLflow Logging
    mlflow.log_param("algorithm", "MiniBatchKMeans")
    mlflow.log_param("n_clusters", K)
    mlflow.log_param("features", ",".join(TEMPORAL_FEATURES))
    mlflow.log_param("sample_size_for_silhouette", 10000)
    mlflow.log_metric("silhouette_score", sil_score)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="temporal_clustering_model",
        registered_model_name="PatrolIQ_Temporal_Clustering_Model"
    )

    print(f"✅ Temporal clustering complete | Silhouette: {sil_score:.3f}")

# ==================================================
# Save Output
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"📁 Saved temporal clustered data → {OUTPUT_FILE}")
