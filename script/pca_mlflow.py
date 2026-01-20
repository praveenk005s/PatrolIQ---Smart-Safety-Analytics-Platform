import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_clustered.csv"

# --------------------------------------------------
# MLflow setup
# --------------------------------------------------
mlflow.set_experiment("PatrolIQ_PCA_Analysis")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(DATA_FILE)

features = [
    "Latitude",
    "Longitude",
    "Hour",
    "Month",
    "Crime_Severity_Score",
    "Arrest_Flag",
    "Domestic_Flag"
]

X = StandardScaler().fit_transform(df[features])

# --------------------------------------------------
# Run PCA experiment
# --------------------------------------------------
with mlflow.start_run(run_name="PCA_2_Components"):
    n_components = 2

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_.sum()

    # 🔹 Log params
    mlflow.log_param("n_components", n_components)
    mlflow.log_param("features_used", len(features))

    # 🔹 Log metrics
    mlflow.log_metric("explained_variance", explained_variance)

    # 🔹 Log model
    mlflow.sklearn.log_model(pca, "pca_model")

    print(f"✅ PCA Explained Variance: {explained_variance:.2%}")
