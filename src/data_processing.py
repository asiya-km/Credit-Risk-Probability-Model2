"""
Data processing and feature engineering pipeline for credit risk modeling.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import os

# Custom transformer for aggregate features
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        agg = X.groupby(self.customer_id_col)[self.amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_amount', 'std')
        ]).reset_index()
        return agg

# Custom transformer for time features
class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col], errors='coerce')
        X['transaction_hour'] = X[self.time_col].dt.hour
        X['transaction_day'] = X[self.time_col].dt.day
        X['transaction_month'] = X[self.time_col].dt.month
        X['transaction_year'] = X[self.time_col].dt.year
        return X

def main():
    # Paths
    raw_path = os.path.join(os.path.dirname(__file__), '../data/raw/data.csv')
    processed_path = os.path.join(os.path.dirname(__file__), '../data/processed/processed_data.csv')

    # Load data
    df = pd.read_csv(raw_path)

    # Feature engineering pipeline
    time_features = ['TransactionStartTime']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    numerical_features = ['Amount', 'Value']

    # Pipelines
    time_pipe = Pipeline([
        ('time_features', TimeFeatures(time_col='TransactionStartTime'))
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('time', time_pipe, time_features),
        ('cat', cat_pipe, categorical_features),
        ('num', num_pipe, numerical_features)
    ], remainder='passthrough')

    # Aggregate features (per customer)
    agg = AggregateFeatures(customer_id_col='CustomerId', amount_col='Amount')
    agg_df = agg.fit_transform(df)

    # Preprocess the main data
    processed = preprocessor.fit_transform(df)
    processed_df = pd.DataFrame(processed)
    processed_df['CustomerId'] = df['CustomerId'].values

    # Get feature names from the preprocessor
    feature_names = []

    # For OneHotEncoder, get feature names
    cat_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    num_features = ['Amount', 'Value']
    time_features = ['TransactionStartTime']

    # Get names for each transformer
    if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
        cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_features)
    else:
        cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(cat_features)

    feature_names.extend(time_features + list(cat_names) + num_features)

    processed_df = pd.DataFrame(processed, columns=feature_names)
    processed_df['CustomerId'] = df['CustomerId'].values

    # Save processed data and aggregate features
    processed_df.to_csv(processed_path, index=False)
    agg_df.to_csv(processed_path.replace('processed_data.csv', 'aggregate_features.csv'), index=False)
    print(f"Processed data saved to {processed_path}")
    print(f"Aggregate features saved to {processed_path.replace('processed_data.csv', 'aggregate_features.csv')}")

if __name__ == "__main__":
    main() 