import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_tsne.csv"

# ==================================================
# Load data
# ==================================================
print("📥 Loading feature-engineered data...")
df = pd.read_csv(INPUT_FILE)

FEATURES = [
    "Latitude",
    "Longitude",
    "Hour",
    "Day_of_Week",
    "Month",
    "Crime_Severity_Score",
    "Arrest_Flag",
    "Domestic_Flag"
]

# ==================================================
# 🔥 SAMPLE (t-SNE MUST be sampled)
# ==================================================
SAMPLE_SIZE = 10_000
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)

X = df_sample[FEATURES]

# ==================================================
# Scaling
# ==================================================
X_scaled = StandardScaler().fit_transform(X)

# ==================================================
# t-SNE
# ==================================================
print("🧠 Running t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42
)

tsne_result = tsne.fit_transform(X_scaled)

# ==================================================
# Save output
# ==================================================
tsne_df = pd.DataFrame(
    tsne_result,
    columns=["TSNE1", "TSNE2"]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
tsne_df.to_csv(OUTPUT_FILE, index=False)

print("✅ t-SNE completed")
print(f"📁 Saved to: {OUTPUT_FILE}")
