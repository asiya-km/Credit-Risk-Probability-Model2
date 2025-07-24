import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def main():
    # Paths
    processed_path = os.path.join(os.path.dirname(__file__), '../data/processed/processed_data.csv')
    raw_path = os.path.join(os.path.dirname(__file__), '../data/raw/data.csv')
    output_path = os.path.join(os.path.dirname(__file__), '../data/processed/processed_with_target.csv')

    # Load data
    df = pd.read_csv(raw_path)
    today = pd.to_datetime(df['TransactionStartTime'], errors='coerce').max() + pd.Timedelta(days=1)

    # RFM calculation
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (today - pd.to_datetime(x, errors='coerce')).min().days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()

    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (lowest Frequency & Monetary, highest Recency)
    cluster_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats.sort_values(['Frequency', 'Monetary', 'Recency'], ascending=[True, True, False]).index[0]
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

    # Merge back to processed data
    processed = pd.read_csv(processed_path)
    processed = processed.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    processed.to_csv(output_path, index=False)
    print(f"Processed data with proxy target saved to {output_path}")

if __name__ == "__main__":
    main()
