import pandas as pd
import mlflow
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_pca.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_tsne_sampled.csv"

# ==================================================
# Load Data
# ==================================================
print("📥 Loading PCA dataset...")
df = pd.read_csv(INPUT_FILE)

FEATURES = ["PCA1", "PCA2", "PCA3"]

# ==================================================
# 🔥 SAMPLE DATA (CRITICAL)
# ==================================================
SAMPLE_SIZE = 10_000
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()

X = df_sample[FEATURES]

# ==================================================
# Scaling
# ==================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================================
# t-SNE
# ==================================================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42
)

# ==================================================
# MLflow
# ==================================================
mlflow.set_experiment("PatrolIQ_tSNE")

with mlflow.start_run(run_name="tSNE_2D_Sampled"):

    print("🧠 Running t-SNE...")
    X_tsne = tsne.fit_transform(X_scaled)

    df_sample["TSNE1"] = X_tsne[:, 0]
    df_sample["TSNE2"] = X_tsne[:, 1]

    mlflow.log_param("perplexity", 30)
    mlflow.log_param("sample_size", SAMPLE_SIZE)

    print("✅ t-SNE completed")

# ==================================================
# Save Output
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_sample.to_csv(OUTPUT_FILE, index=False)

print("💾 t-SNE sampled dataset saved")
print(f"📁 {OUTPUT_FILE}")
