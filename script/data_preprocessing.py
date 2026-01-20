import pandas as pd
from pathlib import Path

# ==================================================
# Paths & Constants
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data/raw/chicago_crime_raw.csv"
PROCESSED_FILE = BASE_DIR / "data/processed/chicago_crime_500k_clean.csv"

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

# ==================================================
# Columns Required for Analysis (memory-safe)
# ==================================================
USE_COLS = [
    "ID", "Case Number","Block", "Primary Type", "Description",
    "Location Description", "Date", "Year",
    "Arrest", "Domestic",
    "Beat", "District", "Ward", "Community Area",
    "Latitude", "Longitude"
]

# ==================================================
# Load & Clean
# ==================================================
def load_raw_data():
    print("📥 Loading raw crime dataset...")
    return pd.read_csv(RAW_FILE, usecols=USE_COLS, low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning data...")

    # Drop duplicates
    df = df.drop_duplicates(subset="ID")

    # Drop missing geo coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Sanity check: Chicago geo bounds
    df = df[
        (df["Latitude"].between(41.6, 42.1)) &
        (df["Longitude"].between(-88.0, -87.4))
    ]

    # Fill missing categorical values
    df["Location Description"] = df["Location Description"].fillna("UNKNOWN")
    df["Ward"] = df["Ward"].fillna(df["Ward"].median())
    df["Community Area"] = df["Community Area"].fillna(df["Community Area"].median())

    # Convert datetime
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # Drop rows with invalid datetime
    df = df.dropna(subset=["Date"])

    return df


def sample_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🎯 Sampling {SAMPLE_SIZE:,} records...")
    if len(df) > SAMPLE_SIZE:
        return df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return df


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    df = load_raw_data()
    print(f"🔢 Raw rows: {len(df):,}")

    df = clean_data(df)
    print(f"✅ After cleaning: {len(df):,}")

    df = sample_data(df)
    print(f"📊 Final dataset size: {len(df):,}")

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    print("🚓 Crime data preprocessing completed successfully")
    print(f"📁 Saved to: {PROCESSED_FILE}")
