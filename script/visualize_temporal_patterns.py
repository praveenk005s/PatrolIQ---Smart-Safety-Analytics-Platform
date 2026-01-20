import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_temporal_clustered.csv"
FIG_DIR = BASE_DIR / "reports/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# ==================================================
# Crimes by Hour
# ==================================================
hourly = df["Hour"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.plot(hourly.index, hourly.values, marker="o")
plt.title("Crimes by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Crimes")
plt.grid(True)

plt.tight_layout()
plt.savefig(FIG_DIR / "crimes_by_hour.png")
plt.close()

# ==================================================
# Temporal Clusters Distribution
# ==================================================
plt.figure(figsize=(8, 5))
df["Temporal_Cluster"].value_counts().sort_index().plot(kind="bar")
plt.title("Temporal Crime Clusters")
plt.xlabel("Cluster")
plt.ylabel("Number of Crimes")

plt.tight_layout()
plt.savefig(FIG_DIR / "temporal_clusters.png")
plt.close()

print("✅ Temporal visualizations saved")
