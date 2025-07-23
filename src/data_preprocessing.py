import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Loads data from a csv file."""
    return pd.read_csv(path)

def preprocess_for_clustering(df, features):
    """Selects features and scales them for clustering."""
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=features), X
