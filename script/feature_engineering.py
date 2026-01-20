import pandas as pd
from pathlib import Path

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clean.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"


# ==================================================
# Temporal Features
# ==================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Hour"] = df["Date"].dt.hour
    df["Day_of_Week"] = df["Date"].dt.dayofweek        # 0 = Monday
    df["Day_Name"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month

    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)

    # Season mapping
    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })

    return df


# ==================================================
# Crime Severity Score
# ==================================================
def add_crime_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Primary Type"] = df["Primary Type"].str.upper()

    severity_map = {
        "HOMICIDE": 5,
        "CRIM SEXUAL ASSAULT": 5,
        "KIDNAPPING": 5,

        "ROBBERY": 4,
        "ASSAULT": 4,
        "WEAPONS VIOLATION": 4,

        "BATTERY": 3,
        "BURGLARY": 3,
        "MOTOR VEHICLE THEFT": 3,

        "THEFT": 2,
        "CRIMINAL DAMAGE": 2,
        "CRIMINAL TRESPASS": 2,

        "NARCOTICS": 1,
        "PUBLIC PEACE VIOLATION": 1
    }

    df["Crime_Severity_Score"] = (
        df["Primary Type"]
        .map(severity_map)
        .fillna(1)
        .astype(int)
    )

    return df


# ==================================================
# Binary Flags
# ==================================================
def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Arrest_Flag"] = df["Arrest"].astype(int)
    df["Domestic_Flag"] = df["Domestic"].astype(int)

    return df


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    print("🚀 Starting Feature Engineering Pipeline...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    print(f"📊 Records loaded: {len(df):,}")

    df = add_temporal_features(df)
    df = add_crime_severity(df)
    df = add_binary_flags(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Feature engineering completed successfully")
    print(f"📁 Saved to: {OUTPUT_FILE}")
