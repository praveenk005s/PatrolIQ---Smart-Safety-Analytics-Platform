# ==================================================
# System path setup (CRITICAL)
# ==================================================
import sys
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ==================================================
# Imports
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np

from script.mlflow_reader import (
    get_best_clustering_run,
    get_pca_run
)

# ==================================================
# Utility: Haversine Distance (meters)
# ==================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="PatrolIQ – Smart Safety Analytics",
    page_icon="🚓",
    layout="wide"
)

# ==================================================
# Load ML model (PIPELINE)
# ==================================================
@st.cache_resource
def load_model():
    model_path = APP_DIR / "models" / "hotspot_model.pkl"
    if not model_path.exists():
        st.error("🚫 Production model not found")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ==================================================
# Paths
# ==================================================
DATA_DIR = BASE_DIR / "data" / "processed"
CLUSTER_FILE = DATA_DIR / "chicago_crime_500k_clustered.csv"
DISTRICT_RISK_FILE = DATA_DIR / "district_risk_map.csv"

# ==================================================
# Load Data
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv(CLUSTER_FILE)
    risk = pd.read_csv(DISTRICT_RISK_FILE)
    return df, risk

df, district_risk = load_data()

# ==================================================
# Detect District Column
# ==================================================
def get_district_column(df):
    for col in ["District", "Community Area", "Ward", "Beat"]:
        if col in df.columns:
            return col
    return None

DISTRICT_COL = get_district_column(df)

# ==================================================
# District → Center
# ==================================================
@st.cache_data
def compute_district_centers(df, col):
    return (
        df.groupby(col)
        .agg(
            center_lat=("Latitude", "mean"),
            center_lon=("Longitude", "mean"),
            crimes=("Latitude", "count")
        )
        .reset_index()
    )

district_centers = (
    compute_district_centers(df, DISTRICT_COL)
    if DISTRICT_COL else None
)

# ==================================================
# Risk lookup
# ==================================================
def get_risk(cluster_id):
    row = district_risk[
        district_risk["Geo_Cluster_KMeans"] == cluster_id
    ]
    return float(row["risk_score"].values[0]) if not row.empty else 0.0

# ==================================================
# Crimes within radius
# ==================================================
def crimes_within_radius(df, lat, lon, radius=1000):
    dist = haversine(
        lat, lon,
        df["Latitude"].values,
        df["Longitude"].values
    )
    return df[dist <= radius]

# ==================================================
# Sidebar
# ==================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "📍 Crime Hotspots",
        "🔮 Real-Time Prediction"
    ]
)

# ==================================================
# 🏠 OVERVIEW
# ==================================================
if page == "🏠 Overview":
    st.title("🚓 PatrolIQ – Smart Safety Analytics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Crimes", f"{len(df):,}")
    c2.metric("Clusters", df["Geo_Cluster_KMeans"].nunique())
    c3.metric(
        "Highest Risk Cluster",
        district_risk.sort_values(
            "risk_score", ascending=False
        )["Geo_Cluster_KMeans"].iloc[0]
    )

# ==================================================
# 📊 EXPLORATORY
# ==================================================
elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Crime Analysis")

    st.subheader("Top Crime Types")
    st.bar_chart(df["Primary Type"].value_counts().head(10))

    st.subheader("District Risk Leaderboard")
    leaderboard = district_risk.sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)
    leaderboard["Rank"] = leaderboard.index + 1
    st.dataframe(leaderboard, use_container_width=True)

# ==================================================
# 📍 HOTSPOTS
# ==================================================
elif page == "📍 Crime Hotspots":
    st.title("📍 Crime Hotspots")

    sample = st.slider("Sample size", 2000, 15000, 6000, 1000)
    map_df = df.sample(sample, random_state=42)

    st.map(
        map_df.rename(
            columns={"Latitude": "lat", "Longitude": "lon"}
        )[["lat", "lon"]]
    )

# ==================================================
# 🔮 REAL-TIME PREDICTION (FIXED)
# ==================================================
elif page == "🔮 Real-Time Prediction":
    st.title("🚨 Patrol Risk Prediction (1 km Radius)")

    # District selector
    if DISTRICT_COL:
        selected_district = st.selectbox(
            "Select District",
            sorted(district_centers[DISTRICT_COL].unique())
        )
        row = district_centers[
            district_centers[DISTRICT_COL] == selected_district
        ].iloc[0]
        default_lat, default_lon = row["center_lat"], row["center_lon"]
    else:
        selected_district = None
        default_lat, default_lon = 41.88, -87.63

    col1, col2, col3 = st.columns(3)
    lat = col1.number_input("Latitude", value=float(default_lat), format="%.6f")
    lon = col2.number_input("Longitude", value=float(default_lon), format="%.6f")
    severity = col3.slider("Expected Severity", 1, 5, 3)

    if st.button("Analyze Area"):
        scope_df = (
            df[df[DISTRICT_COL] == selected_district]
            if DISTRICT_COL else df
        )

        nearby = crimes_within_radius(scope_df, lat, lon, 1000)

        input_df = pd.DataFrame(
            [[lat, lon, severity]],
            columns=["Latitude", "Longitude", "Crime_Severity_Score"]
        )

        cluster = int(model.predict(input_df)[0])
        risk = get_risk(cluster)

        st.success(f"Predicted Cluster: {cluster}")
        st.metric("Crimes in 1 km", len(nearby))
        st.metric("Risk Score", f"{risk:.1f}/100")

        st.map(pd.concat([
            nearby.rename(
                columns={"Latitude": "lat", "Longitude": "lon"}
            )[["lat", "lon"]],
            pd.DataFrame({"lat": [lat], "lon": [lon]})
        ]))




