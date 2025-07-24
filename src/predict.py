"""
Inference script for credit risk modeling.
"""
import pandas as pd
import mlflow
import os

def main():
    # Path to processed data (for feature columns)
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/processed_with_target.csv')
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in ['is_high_risk', 'CustomerId']]

    # Example: create a sample input (replace with real data as needed)
    sample = df[feature_cols].iloc[[0]].copy()  # Use the first row as a sample
    print("Sample input:")
    print(sample)

    # Load best model from MLflow Model Registry
    model_uri = "models:/credit-risk-best-model/Production"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Could not load model from MLflow: {e}\nTrying to load the latest version instead.")
        model_uri = "models:/credit-risk-best-model/1"
        model = mlflow.sklearn.load_model(model_uri)

    # Predict risk probability
    risk_proba = model.predict_proba(sample)[:, 1][0]
    print(f"Predicted risk probability: {risk_proba:.4f}")

if __name__ == "__main__":
    main() 