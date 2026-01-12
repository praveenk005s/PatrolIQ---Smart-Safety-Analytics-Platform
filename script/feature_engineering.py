import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_clean.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_features.csv"

def add_temporal_features(df):
    df = df.copy()

    df["Hour"] = df["Date"].dt.hour
    df["Day_of_Week"] = df["Date"].dt.dayofweek      # 0 = Monday
    df["Month"] = df["Date"].dt.month
    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)

    return df

def add_crime_severity(df):
    df = df.copy()

    severity_map = {
        "HOMICIDE": 5,
        "CRIM SEXUAL ASSAULT": 5,
        "ROBBERY": 4,
        "ASSAULT": 4,
        "BATTERY": 3,
        "BURGLARY": 3,
        "MOTOR VEHICLE THEFT": 3,
        "THEFT": 2,
        "CRIMINAL DAMAGE": 2,
        "NARCOTICS": 1
    }

    df["Crime_Severity_Score"] = df["Primary Type"].map(severity_map).fillna(1)

    return df

def add_binary_flags(df):
    df = df.copy()

    df["Arrest_Flag"] = df["Arrest"].astype(int)
    df["Domestic_Flag"] = df["Domestic"].astype(int)

    return df

if __name__ == "__main__":
    print("🚀 Starting Feature Engineering...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    print(f"📊 Records loaded: {len(df)}")

    df = add_temporal_features(df)
    df = add_crime_severity(df)
    df = add_binary_flags(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Feature engineering completed successfully")
    print(f"📁 Saved to: {OUTPUT_FILE}")
