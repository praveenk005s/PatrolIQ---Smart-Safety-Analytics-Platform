import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clustered.csv"
FIG_DIR = BASE_DIR / "reports/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Load data
# ==================================================
df = pd.read_csv(INPUT_FILE)

# ==================================================
# Scatter plot: Crime Hotspots
# ==================================================
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    df["Longitude"],
    df["Latitude"],
    c=df["Geo_Cluster_KMeans"],
    cmap="tab10",
    s=1,
    alpha=0.4
)

plt.colorbar(scatter, label="Crime Cluster")
plt.title("Geographic Crime Hotspots (KMeans)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig(FIG_DIR / "crime_hotspots_kmeans.png")
plt.close()

print("✅ Crime hotspot map saved")
