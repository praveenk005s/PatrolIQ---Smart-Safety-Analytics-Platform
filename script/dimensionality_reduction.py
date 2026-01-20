import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_pca.csv"

# ==================================================
# Load Data
# ==================================================
print("📥 Loading feature-engineered data...")
df = pd.read_csv(INPUT_FILE)

FEATURES = [
    "Latitude",
    "Longitude",
    "Hour",
    "Day_of_Week",
    "Month",
    "Is_Weekend",
    "Crime_Severity_Score",
    "Arrest_Flag",
    "Domestic_Flag"
]

X = df[FEATURES]

# ==================================================
# Scaling
# ==================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================================
# PCA
# ==================================================
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ==================================================
# MLflow Experiment
# ==================================================
mlflow.set_experiment("PatrolIQ_PCA")

with mlflow.start_run(run_name="PCA_3_Components"):

    explained_var = pca.explained_variance_ratio_.sum()

    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]
    df["PCA3"] = X_pca[:, 2]

    # Log parameters
    mlflow.log_param("n_components", 3)
    mlflow.log_param("features_used", ",".join(FEATURES))

    # Log metrics
    mlflow.log_metric("explained_variance_ratio", explained_var)

    # Log model
    mlflow.sklearn.log_model(pca, "pca_model")

    print(f"✅ PCA Explained Variance: {explained_var:.2%}")

# ==================================================
# Save Output
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("💾 PCA reduced dataset saved")
print(f"📁 {OUTPUT_FILE}")
