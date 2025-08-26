# Mall Customer Clustering - evaluate.py

import pandas as pd
import joblib

# 1. Load saved model and dataset
model = joblib.load(r"C:\mall_customer2\kmeans_model.pkl")
df = pd.read_csv(r"C:\mall_customer2\Mall_Customers_with_clusters.csv")

# 2. Show basic cluster stats
print("✅ Model and clustered data loaded successfully!\n")

print("📊 Number of customers per cluster:")
print(df["Cluster"].value_counts().sort_index())
print()

print("📌 Cluster centroids (Income, SpendingScore):")
print(model.cluster_centers_)
print()

# 3. Example: look at first few clustered customers
print("👀 Sample of clustered customers:")
print(df.head(10))
