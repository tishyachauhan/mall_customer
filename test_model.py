# test_model.py

import joblib

def predict_cluster(income, spending_score):
    # Load trained model
    model = joblib.load("kmeans_model.pkl")
    
    # Prepare input
    X_new = [[income, spending_score]]
    
    # Predict cluster
    cluster = model.predict(X_new)[0]
    return cluster


if __name__ == "__main__":
    # Example: user input
    try:
        income = float(input("Enter Annual Income (in k$): "))
        spending = float(input("Enter Spending Score (1-100): "))
        
        cluster = predict_cluster(income, spending)
        print(f"Predicted Cluster: {cluster}")
    except Exception as e:
        print("Error:", e)
