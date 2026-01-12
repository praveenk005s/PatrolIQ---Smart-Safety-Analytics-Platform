import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chicago_crime_500k_clustered.csv"

def run_kmeans(df, k=6):
    print("📍 Selecting features...")
    X = df[["Latitude", "Longitude", "Crime_Severity_Score"]]

    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🚀 Running MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=10_000,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)
    df["Geo_Cluster_KMeans"] = labels

    # ✅ Silhouette score on SAMPLE only
    print("📊 Calculating silhouette score on sample...")
    sample_size = 10_000

    X_sample, labels_sample = resample(
        X_scaled,
        labels,
        n_samples=sample_size,
        random_state=42
    )

    score = silhouette_score(X_sample, labels_sample)
    print(f"✅ KMeans Silhouette Score (sampled): {score:.3f}")

    return df

if __name__ == "__main__":
    print("🚓 Running Crime Hotspot Clustering (FAST VERSION)...")

    df = pd.read_csv(INPUT_FILE)
    df = run_kmeans(df, k=6)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("📁 Clustered data saved successfully")
