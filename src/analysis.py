# src/analysis.py

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import *
from data_preprocessing import load_data, preprocess_for_clustering

def run_clustering_analysis(X_scaled, X_original):
    """Runs different clustering algorithms and returns their labels."""
    # K-Means
    kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # DBSCAN
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Agglomerative Clustering
    agg_cluster = AgglomerativeClustering(n_clusters=KMEANS_N_CLUSTERS)
    agg_labels = agg_cluster.fit_predict(X_scaled)
    
    return {
        "K-Means": kmeans_labels,
        "DBSCAN": dbscan_labels,
        "Agglomerative": agg_labels
    }

def evaluate_clusters(X_scaled, labels_dict):
    """Evaluates clustering results using silhouette and Davies-Bouldin scores."""
    metrics = []
    for name, labels in labels_dict.items():
        # Metrics can only be calculated if more than 1 cluster is found
        if len(set(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # -1 is noise in DBSCAN
        else:
            sil_score, db_score, n_clusters = None, None, 1
            
        metrics.append({
            "Algorithm": name,
            "Silhouette Score": sil_score,
            "Davies-Bouldin Index": db_score,
            "Number of Clusters": n_clusters
        })
    return pd.DataFrame(metrics)

def plot_and_save_clusters(X_original, labels_dict, features, save_path):
    """Creates and saves scatter plots for each clustering result."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for name, labels in labels_dict.items():
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=X_original[features[0]],
            y=X_original[features[1]],
            hue=labels,
            palette=sns.color_palette("hsv", len(set(labels)))
        )
        plt.title(f'Clusters found by {name}')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend(title="Cluster")
        
        filepath = os.path.join(save_path, f"{name.lower()}_clusters.png")
        plt.savefig(filepath)
        print(f"Plot saved to {filepath}")
        plt.close()

def main():
    """Main function to run the full analysis pipeline."""
    df = load_data(DATA_PATH)
    X_scaled, X_original = preprocess_for_clustering(df, FEATURES)
    
    labels_dict = run_clustering_analysis(X_scaled, X_original)
    
    metrics_df = evaluate_clusters(X_scaled, labels_dict)
    print("--- Clustering Evaluation Metrics ---")
    print(metrics_df)
    
    # Save metrics
    if not os.path.exists(os.path.dirname(RESULTS_METRICS_PATH)):
        os.makedirs(os.path.dirname(RESULTS_METRICS_PATH))
    metrics_df.to_csv(RESULTS_METRICS_PATH, index=False)
    print(f"\nMetrics saved to {RESULTS_METRICS_PATH}")
    
    # Create and save visualizations
    plot_and_save_clusters(X_original, labels_dict, FEATURES, VISUALIZATIONS_PATH)

if __name__ == "__main__":
    main()