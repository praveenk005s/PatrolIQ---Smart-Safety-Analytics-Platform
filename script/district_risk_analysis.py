import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(DATA_DIR / "chicago_crime_500k_clustered.csv")

district_risk = (
    df.groupby("District")
      .agg(
          total_crimes=("ID", "count"),
          avg_severity=("Crime_Severity_Score", "mean"),
          dominant_cluster=("Geo_Cluster_KMeans", lambda x: x.value_counts().idxmax())
      )
      .reset_index()
)

# Normalize risk score
district_risk["risk_score"] = (
    district_risk["total_crimes"].rank(pct=True) * 100
)

district_risk.to_csv(
    DATA_DIR / "district_risk_map.csv",
    index=False
)

print("✅ District-wise risk file created")
