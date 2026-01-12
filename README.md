🚓 PatrolIQ – Smart Safety Analytics Platform
📌 Project Overview

PatrolIQ is an end-to-end smart safety analytics platform that analyzes large-scale urban crime data using unsupervised machine learning.
The system identifies crime hotspots, uncovers spatio-temporal patterns, and provides interactive visualizations through a Streamlit web application.

This project demonstrates real-world data engineering, machine learning, MLOps (MLflow), and cloud deployment skills.

🎯 Problem Statement

Urban crime data is large, noisy, and complex.
Traditional rule-based analysis fails to reveal hidden spatial and temporal crime patterns.

Goal:
Build a scalable analytics system to:

Identify crime hotspots

Understand spatial risk zones

Visualize crime patterns

Track ML experiments

Deploy insights via a web dashboard

🧠 Solution Approach
✔ Data Processing

Cleaned and preprocessed ~500K crime records

Handled missing geo-coordinates and categorical data

Engineered time-based and severity features

✔ Feature Engineering

Crime severity scoring

Temporal features (hour, month)

Binary flags (arrest, domestic)

✔ Unsupervised Learning

MiniBatch KMeans for crime hotspot clustering

Optimized for large datasets

Cluster evaluation using Silhouette Score

✔ Dimensionality Reduction

PCA for linear pattern visualization

t-SNE (sample-based) for non-linear structure discovery

✔ MLOps

MLflow used for:

Experiment tracking

Parameter logging

Metric comparison

Model versioning

✔ Visualization & Deployment

Streamlit interactive dashboard

Deployed on Streamlit Cloud

Modular, production-ready structure


📊 Streamlit Dashboard Features

Overview

Total crimes

Cluster count

High-risk zones

Exploratory Data Analysis

Top crime types

Hourly crime trends

Crime Hotspot Map

Geo-spatial clustering visualization

PCA Visualization

Linear crime pattern separation

t-SNE Visualization

Non-linear crime pattern insights (sample-based)

MLflow Experiments

Best clustering model

PCA explained variance

🧪 Model Performance
Model	Metric
MiniBatch KMeans	Silhouette Score ≈ 0.30
PCA	Explained Variance ≈ 40%

📌 Note: Lower silhouette scores are expected for noisy geo-spatial crime data and still provide operational insights.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib

MLflow

Streamlit

Git & GitHub

Streamlit Cloud

🚀 How to Run Locally
# Clone repository
git clone https://github.com/praveenk005s/PatrolIQ---Smart-Safety-Analytics-Platform.git
cd PatrolIQ---Smart-Safety-Analytics-Platform

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app/app.py

☁️ Cloud Deployment

The application is deployed using Streamlit Cloud, enabling public access and easy sharing.

📈 Key Learnings

Handling large real-world datasets

Unsupervised ML for spatial analytics

Experiment tracking with MLflow

Performance optimization using sampling & caching

End-to-end ML system deployment



👤 Author

Praveen Kumar
Aspiring Data Scientist | Machine Learning Engineer

🔗 GitHub: https://github.com/praveenk005s

⭐ Acknowledgement

Crime dataset inspired by public urban crime data sources.

