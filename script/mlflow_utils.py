import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_features.csv"

# --------------------------------------------------
# MLflow setup
# --------------------------------------------------
mlflow.set_experiment("PatrolIQ_Crime_Clustering")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(DATA_FILE)

features = ["Latitude", "Longitude", "Crime_Severity_Score"]
X = df[features]

X_scaled = StandardScaler().fit_transform(X)

# --------------------------------------------------
# Run experiment
# --------------------------------------------------
with mlflow.start_run(run_name="MiniBatchKMeans_K6"):
    k = 6

    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=10_000,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    # Silhouette on sample
    X_sample, labels_sample = resample(
        X_scaled, labels,
        n_samples=10_000,
        random_state=42
    )

    score = silhouette_score(X_sample, labels_sample)

    # 🔹 Log params
    mlflow.log_param("algorithm", "MiniBatchKMeans")
    mlflow.log_param("n_clusters", k)
    mlflow.log_param("dataset_size", len(df))

    # 🔹 Log metrics
    mlflow.log_metric("silhouette_score", score)

    # 🔹 Log model
    mlflow.sklearn.log_model(model, "clustering_model")

    print(f"✅ MLflow Run Logged | Silhouette Score: {score:.3f}")
