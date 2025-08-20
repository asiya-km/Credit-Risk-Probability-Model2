"""
Model explainability using SHAP for credit risk probability model.
"""
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def plot_global_shap(model: BaseEstimator, X: pd.DataFrame, out_path='shap_global.png'):
    """
    Generate and save SHAP summary plot (global feature importance).
    """
    explainer = None
    if hasattr(model, 'predict_proba'):
        try:
            explainer = shap.Explainer(model, X)
        except Exception:
            explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP global summary plot saved to {out_path}")


def plot_local_shap(model: BaseEstimator, X: pd.DataFrame, out_path_pattern='shap_local_{i}.png', indices=None):
    """
    Generate and save SHAP force plots for selected samples (local explanations).
    indices: List of row indices for which to generate force plots.
    """
    if indices is None:
        indices = [0]
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    for i in indices:
        plt.figure()
        shap.plots.force(shap_values[i], matplotlib=True, show=False)
        file_path = out_path_pattern.format(i=i)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        print(f"SHAP local force plot for sample {i} saved to {file_path}")
