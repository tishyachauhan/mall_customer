# Mall Customer Clustering - Minimal train.py

import pandas as pd
from sklearn.cluster import KMeans
import joblib   # for saving the model

# 1. Load dataset
df = pd.read_csv(r"C:\mall_customer\data\Mall_Customers.csv")

# 2. Rename columns (clean format)
df = df.rename(columns={
    "Genre": "Gender",
    "Age": "Age",
    "Annual Income (k$)": "Income",
    "Spending Score (1-100)": "SpendingScore"
})

# 3. Select features for clustering
X = df[["Income", "SpendingScore"]]

# 4. Train final model (k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# 5. Save results and model
df.to_csv(r"C:\mall_customer2\Mall_Customers_with_clusters.csv", index=False)
joblib.dump(kmeans, r"C:\mall_customer2\kmeans_model.pkl")

