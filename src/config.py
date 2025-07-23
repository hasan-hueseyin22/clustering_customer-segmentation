# src/config.py

# Paths
DATA_PATH = "data/raw/Mall_Customers.csv"
RESULTS_METRICS_PATH = "results/comparison_metrics.csv"
VISUALIZATIONS_PATH = "visualizations/"

# Features to use for clustering
# We focus on income and spending score for clear 2D visualization
FEATURES = ['Annual Income (k$)', 'Spending Score (1-100)']

# K-Means settings
KMEANS_N_CLUSTERS = 5

# DBSCAN settings
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5