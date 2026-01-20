import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_clean.csv"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
print(f"✅ Records loaded: {len(df)}")

# --------------------------------------------------
# 1. Basic Overview
# --------------------------------------------------
print("\n📊 Dataset Info")
print(df.info())

print("\n❓ Missing Values")
missing = df.isna().sum()
missing[missing > 0].to_csv(REPORT_DIR / "missing_values_summary.csv")

# --------------------------------------------------
# 2. Crime Type Distribution
# --------------------------------------------------
print("\n🚔 Top Crime Types")
crime_counts = df["Primary Type"].value_counts().head(10)
crime_counts.to_csv(REPORT_DIR / "top_crime_types.csv")

plt.figure(figsize=(10, 5))
crime_counts.plot(kind="bar")
plt.title("Top 10 Crime Types")
plt.xlabel("Crime Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "top_10_crime_types.png")
plt.close()

# --------------------------------------------------
# 3. Temporal Analysis
# --------------------------------------------------
df["Hour"] = df["Date"].dt.hour
hourly_counts = df["Hour"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
hourly_counts.plot(kind="line", marker="o")
plt.title("Crimes by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "crimes_by_hour.png")
plt.close()

# --------------------------------------------------
# 4. Weekday vs Weekend
# --------------------------------------------------
df["Is_Weekend"] = df["Date"].dt.dayofweek.isin([5, 6])
weekend_counts = df["Is_Weekend"].value_counts()

weekend_counts.to_csv(REPORT_DIR / "weekend_vs_weekday.csv")

# --------------------------------------------------
# 5. Arrest & Domestic Analysis
# --------------------------------------------------
arrest_rate = df["Arrest"].value_counts(normalize=True) * 100
domestic_rate = df["Domestic"].value_counts(normalize=True) * 100

arrest_rate.to_csv(REPORT_DIR / "arrest_rate.csv")
domestic_rate.to_csv(REPORT_DIR / "domestic_rate.csv")

# --------------------------------------------------
# 6. Geographic Summary
# --------------------------------------------------
geo_summary = df[["Latitude", "Longitude"]].describe()
geo_summary.to_csv(REPORT_DIR / "geo_summary.csv")

print("\n✅ Exploratory Analysis Completed Successfully")
print(f"📁 Reports saved to: {REPORT_DIR}")
print(f"📊 Figures saved to: {FIGURE_DIR}")
