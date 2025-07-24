"""
Model training script for credit risk modeling.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os

def main():
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/processed_with_target.csv')
    df = pd.read_csv(data_path)

    # Features and target
    X = df.drop(columns=['is_high_risk', 'CustomerId'])
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df['is_high_risk']

    # Debug: print dtypes if needed
    # print(X.dtypes)
    # print(X.head())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Models and hyperparameters
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42)
    }
    param_grids = {
        'LogisticRegression': {'C': [0.1, 1, 10]},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    }

    best_model = None
    best_score = 0
    best_name = ''

    mlflow.set_experiment('credit-risk-model')

    for name, model in models.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        print(f"Best params: {grid.best_params_}")
        print(f"Metrics: {metrics}")

        # MLflow logging
        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(grid.best_estimator_, name)

        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model = grid.best_estimator_
            best_name = name

    print(f"\nBest model: {best_name} (ROC-AUC: {best_score:.4f})")
    # Register best model
    with mlflow.start_run(run_name=f"register_{best_name}"):
        mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="credit-risk-best-model")
        print("Best model registered in MLflow.")

if __name__ == "__main__":
    main() 