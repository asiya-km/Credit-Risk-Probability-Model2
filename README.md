# Credit Risk Probability Model for Alternative Data

## Project Overview
This project aims to build a credit scoring model for Bati Bank's buy-now-pay-later service, leveraging eCommerce behavioral data to predict customer credit risk.

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
Basel II requires financial institutions to rigorously measure, document, and manage credit risk, emphasizing transparency, auditability, and regulatory compliance. This means our credit risk model must be interpretable, so that risk drivers and predictions can be clearly explained to regulators, auditors, and business stakeholders. Well-documented models facilitate validation, monitoring, and updates, ensuring the bank meets regulatory standards and can justify lending decisions.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Without a direct default label, we must engineer a proxy (e.g., using RFM analysis to identify disengaged customers as high risk) to train our model. This is necessary to enable supervised learning. However, using a proxy introduces risks: the proxy may not perfectly represent true default behavior, potentially leading to misclassification, biased decisions, or regulatory scrutiny if the model's predictions are not aligned with actual credit outcomes. Careful validation and business oversight are required to mitigate these risks.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Simple models (e.g., Logistic Regression with WoE) offer high interpretability, easier validation, and regulatory acceptance, but may sacrifice predictive power. Complex models (e.g., Gradient Boosting) can achieve higher accuracy but are less transparent, harder to explain, and may face regulatory resistance. In regulated contexts, the trade-off is between maximizing performance and ensuring the model is explainable, auditable, and compliant with risk management standards.

## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
``` 