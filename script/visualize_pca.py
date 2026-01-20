import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

PCA_FILE = BASE_DIR / "data/processed/chicago_crime_500k_pca.csv"
CLUSTER_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clustered.csv"

FIG_DIR = BASE_DIR / "reports/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Load data
# ==================================================
df_pca = pd.read_csv(PCA_FILE)
df_cluster = pd.read_csv(CLUSTER_FILE)

# ==================================================
# Merge PCA + Cluster labels
# ==================================================
df = pd.concat(
    [
        df_pca[["PCA1", "PCA2"]],
        df_cluster["Geo_Cluster_KMeans"]
    ],
    axis=1
)

# ==================================================
# Plot
# ==================================================
plt.figure(figsize=(9, 7))
plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Geo_Cluster_KMeans"],
    cmap="tab10",
    s=2,
    alpha=0.5
)

plt.title("PCA Visualization of Crime Clusters (KMeans)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Crime Cluster")

plt.tight_layout()
plt.savefig(FIG_DIR / "pca_clusters.png")
plt.close()

print("✅ PCA + KMeans cluster visualization saved")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

PCA_FILE = BASE_DIR / "data/processed/chicago_crime_500k_pca.csv"
CLUSTER_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clustered.csv"

FIG_DIR = BASE_DIR / "reports/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Load data
# ==================================================
df_pca = pd.read_csv(PCA_FILE)
df_cluster = pd.read_csv(CLUSTER_FILE)

# ==================================================
# Merge PCA + Cluster labels
# ==================================================
df = pd.concat(
    [
        df_pca[["PCA1", "PCA2"]],
        df_cluster["Geo_Cluster_KMeans"]
    ],
    axis=1
)

# ==================================================
# Plot
# ==================================================
plt.figure(figsize=(9, 7))
plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Geo_Cluster_KMeans"],
    cmap="tab10",
    s=2,
    alpha=0.5
)

plt.title("PCA Visualization of Crime Clusters (KMeans)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Crime Cluster")

plt.tight_layout()
plt.savefig(FIG_DIR / "pca_clusters.png")
plt.close()

print("✅ PCA + KMeans cluster visualization saved")
