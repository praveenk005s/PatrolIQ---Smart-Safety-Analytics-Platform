import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/processed/chicago_crime_500k_features.csv"

FEATURES = [
    "Latitude", "Longitude", "Hour", "Day_of_Week",
    "Month", "Is_Weekend", "Crime_Severity_Score",
    "Arrest_Flag", "Domestic_Flag"
]

df = pd.read_csv(INPUT_FILE)
X = df[FEATURES]

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
pca.fit(X_scaled)

importance = pd.DataFrame(
    pca.components_.T,
    columns=["PCA1", "PCA2", "PCA3"],
    index=FEATURES
)

print("\n📊 Feature Importance (PCA Loadings):")
print(importance.abs().sort_values("PCA1", ascending=False))
