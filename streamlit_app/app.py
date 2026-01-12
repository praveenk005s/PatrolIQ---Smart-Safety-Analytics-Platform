import sys
from pathlib import Path

# --------------------------------------------------
# Add PatrolIQ root to PYTHONPATH
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# --------------------------------------------------
# Imports
# --------------------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from script.mlflow_reader import (
    get_best_clustering_run,
    get_pca_run
)



# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="PatrolIQ – Smart Safety Analytics",
    page_icon="🚓",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------


DATA_DIR = BASE_DIR / "data" / "processed"

CLUSTER_FILE = DATA_DIR / "chicago_crime_500k_clustered.csv"
REDUCED_FILE = DATA_DIR / "chicago_crime_500k_reduced.csv"

# --------------------------------------------------
# Load data (cached)
# --------------------------------------------------
@st.cache_data
def load_data():
    df_clustered = pd.read_csv(CLUSTER_FILE)
    df_reduced = pd.read_csv(REDUCED_FILE)
    return df_clustered, df_reduced

df, df_reduced = load_data()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "📍 Crime Hotspots",
        "📉 PCA Visualization",
        "🧠 t-SNE Visualization",
        "📈 MLflow Experiments"
    ]
)


# --------------------------------------------------
# 🏠 Overview
# --------------------------------------------------
if page == "🏠 Overview":
    st.title("🚓 PatrolIQ – Smart Safety Analytics Platform")

    st.markdown("""
    **PatrolIQ** analyzes large-scale urban crime data using  
    **unsupervised machine learning** to identify hotspots,  
    temporal patterns, and spatial risk zones.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Crimes", f"{len(df):,}")
    col2.metric("Crime Clusters", df["Geo_Cluster_KMeans"].nunique())
    col3.metric("High-Risk Zones", df["Geo_Cluster_KMeans"].value_counts().idxmax())

# --------------------------------------------------
# 📊 Exploratory Analysis
# --------------------------------------------------
elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Crime Analysis")

    st.subheader("Top Crime Types")
    crime_counts = df["Primary Type"].value_counts().head(10)
    st.bar_chart(crime_counts)

    st.subheader("Crimes by Hour")
    df["Hour"] = pd.to_datetime(df["Date"], errors="coerce").dt.hour
    hourly = df["Hour"].value_counts().sort_index()
    st.line_chart(hourly)

# --------------------------------------------------
# 📍 Crime Hotspots
# --------------------------------------------------
elif page == "📍 Crime Hotspots":
    st.title("📍 Crime Hotspot Map")

    st.markdown("Crime clusters identified using **MiniBatch KMeans**")

    sample_size = st.slider(
        "Select sample size for map",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

    map_df = df.sample(sample_size, random_state=42)

    
    map_data = map_df.rename(
    columns={"Latitude": "lat", "Longitude": "lon"}
    )

    st.map(map_data[["lat", "lon"]])


    st.subheader("Cluster Distribution")
    st.bar_chart(map_df["Geo_Cluster_KMeans"].value_counts())

# --------------------------------------------------
# 📉 PCA Visualization
# --------------------------------------------------
elif page == "📉 PCA Visualization":
    st.title("📉 PCA – Crime Pattern Visualization")

    st.markdown("Linear dimensionality reduction using **PCA**")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df_reduced["PCA1"],
        df_reduced["PCA2"],
        c=df_reduced["Geo_Cluster_KMeans"],
        s=2,
        alpha=0.5
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA Crime Clusters")

    st.pyplot(fig)

# --------------------------------------------------
# 🧠 t-SNE Visualization
# --------------------------------------------------
elif page == "🧠 t-SNE Visualization":
    st.title("🧠 t-SNE – Non-Linear Crime Patterns")

    st.markdown("""
    t-SNE applied on a **10,000 sample**  
    to visualize complex, non-linear relationships.
    """)

    if "TSNE1" not in df_reduced.columns:
        st.warning("t-SNE data available only for sampled points.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            df_reduced["TSNE1"],
            df_reduced["TSNE2"],
            c=df_reduced["Geo_Cluster_KMeans"],
            s=4,
            alpha=0.6
        )

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title("t-SNE Crime Clusters")

        st.pyplot(fig)
elif page == "📈 MLflow Experiments":
    st.title("📈 MLflow Experiment Tracking")

    st.subheader("Best Crime Clustering Model")
    cluster_run = get_best_clustering_run()

    if cluster_run:
        st.metric("Silhouette Score", f"{cluster_run['silhouette_score']:.3f}")
        st.write("Algorithm:", cluster_run["algorithm"])
        st.write("Clusters:", cluster_run["n_clusters"])
    else:
        st.warning("No clustering runs found")

    st.divider()

    st.subheader("Best PCA Model")
    pca_run = get_pca_run()

    if pca_run:
        st.metric(
            "Explained Variance",
            f"{pca_run['explained_variance']:.2%}"
        )
        st.write("Components:", pca_run["n_components"])
    else:
        st.warning("No PCA runs found")
