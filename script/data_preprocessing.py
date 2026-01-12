import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "chicago_crime_raw.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_clean.csv"

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42   # for reproducibility

def load_and_sample_data():
    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_FILE)

    print(f"🔢 Total rows in raw data: {len(df)}")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"🎯 Sampled {SAMPLE_SIZE} rows")
    else:
        print("⚠️ Dataset has less than 500k rows, using full data")

    return df

def clean_data(df):
    # Drop rows without geo coordinates
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    # Fill missing values
    df.loc[:, "Location Description"] = df["Location Description"].fillna("UNKNOWN")
    df.loc[:, "Ward"] = df["Ward"].fillna(df["Ward"].median())
    df.loc[:, "Community Area"] = df["Community Area"].fillna(df["Community Area"].median())

    # Convert datetime safely
    df.loc[:, "Date"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    return df

if __name__ == "__main__":
    df = load_and_sample_data()
    df = clean_data(df)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    print("✅ 500K crime records sampled & cleaned successfully")
    print(f"📁 Saved to: {PROCESSED_FILE}")
