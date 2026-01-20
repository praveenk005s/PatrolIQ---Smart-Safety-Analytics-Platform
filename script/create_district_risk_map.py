import pandas as pd
from pathlib import Path

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clustered.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/district_risk_map.csv"

# ==================================================
# Load clustered data
# ==================================================
print("📥 Loading clustered crime data...")
df = pd.read_csv(INPUT_FILE)

# ==================================================
# Validate
# ==================================================
required_cols = {"Geo_Cluster_KMeans", "Crime_Severity_Score"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"❌ Missing columns: {missing}")

# ==================================================
# Compute risk per cluster
# ==================================================
print("🔥 Computing risk scores per cluster...")

risk_df = (
    df.groupby("Geo_Cluster_KMeans")
      .agg(
          crime_count=("Geo_Cluster_KMeans", "size"),
          avg_severity=("Crime_Severity_Score", "mean")
      )
      .reset_index()
)

# ==================================================
# Risk Score Formula (Explainable & Interview-safe)
# ==================================================
# risk_score = normalized(crime_count * avg_severity)
risk_df["raw_risk"] = (
    risk_df["crime_count"] * risk_df["avg_severity"]
)

risk_df["risk_score"] = (
    100 * risk_df["raw_risk"] / risk_df["raw_risk"].max()
)

risk_df = risk_df.sort_values("risk_score", ascending=False)

# ==================================================
# Save
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
risk_df[["Geo_Cluster_KMeans", "risk_score"]].to_csv(
    OUTPUT_FILE,
    index=False
)

print("✅ district_risk_map.csv generated successfully")
print(f"📁 Saved at: {OUTPUT_FILE}")
