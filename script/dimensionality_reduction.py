import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_clustered.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_reduced.csv"
FIGURE_DIR = BASE_DIR / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("📥 Loading clustered dataset...")
df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# Features for reduction
# --------------------------------------------------
features = [
    "Latitude",
    "Longitude",
    "Hour",
    "Month",
    "Crime_Severity_Score",
    "Arrest_Flag",
    "Domestic_Flag"
]

X = df[features]

# --------------------------------------------------
# Scaling
# --------------------------------------------------
print("⚖️ Scaling features...")
X_scaled = StandardScaler().fit_transform(X)

# --------------------------------------------------
# PCA
# --------------------------------------------------
print("📉 Applying PCA...")
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(X_scaled)

df["PCA1"] = pca_components[:, 0]
df["PCA2"] = pca_components[:, 1]

print(f"✅ PCA Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")

# --------------------------------------------------
# PCA Plot
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Geo_Cluster_KMeans"],
    s=1,
    alpha=0.5
)
plt.title("PCA – Crime Cluster Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "pca_clusters.png")
plt.close()

# --------------------------------------------------
# t-SNE (on SAMPLE – very important)
# --------------------------------------------------
print("🧠 Applying t-SNE on sample...")
sample_df = df.sample(10_000, random_state=42)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42
)

tsne_components = tsne.fit_transform(
    StandardScaler().fit_transform(sample_df[features])
)

sample_df["TSNE1"] = tsne_components[:, 0]
sample_df["TSNE2"] = tsne_components[:, 1]

# --------------------------------------------------
# t-SNE Plot
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    sample_df["TSNE1"],
    sample_df["TSNE2"],
    c=sample_df["Geo_Cluster_KMeans"],
    s=5,
    alpha=0.6
)
plt.title("t-SNE – Crime Pattern Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "tsne_clusters.png")
plt.close()




# --------------------------------------------------
# Save reduced dataset
# --------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Dimensionality reduction completed")
print(f"📁 Reduced dataset saved to: {OUTPUT_FILE}")
print(f"📊 Plots saved to: {FIGURE_DIR}")
