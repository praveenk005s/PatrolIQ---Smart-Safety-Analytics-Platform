import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

TSNE_FILE = BASE_DIR / "data/processed/chicago_crime_500k_tsne.csv"
CLUSTER_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clustered.csv"

FIG_DIR = BASE_DIR / "reports/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Load data
# ==================================================
df_tsne = pd.read_csv(TSNE_FILE)
df_cluster = pd.read_csv(CLUSTER_FILE)

# ==================================================
# Merge t-SNE + KMeans labels
# ==================================================
df = pd.concat(
    [
        df_tsne[["TSNE1", "TSNE2"]],
        df_cluster["Geo_Cluster_KMeans"]
    ],
    axis=1
)

# ==================================================
# Plot t-SNE
# ==================================================
plt.figure(figsize=(9, 7))
plt.scatter(
    df["TSNE1"],
    df["TSNE2"],
    c=df["Geo_Cluster_KMeans"],
    cmap="tab10",
    s=5,
    alpha=0.6
)

plt.title("t-SNE Visualization of Crime Clusters (KMeans)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(label="Crime Cluster")

plt.tight_layout()
plt.savefig(FIG_DIR / "tsne_clusters.png")
plt.close()

print("✅ t-SNE + KMeans cluster visualization saved")
