import pandas as pd
import numpy as np
import pytest
from src.data_processing import AggregateFeatures, TimeFeatures

def test_aggregate_features_basic():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 2],
        'Amount': [100, 200, 150, 50, 300]
    })
    agg = AggregateFeatures(customer_id_col='CustomerId', amount_col='Amount')
    out = agg.fit_transform(df)
    assert 'total_amount' in out.columns
    assert 'avg_amount' in out.columns
    assert 'transaction_count' in out.columns
    assert out[out['CustomerId']==1]['total_amount'].iloc[0] == 300
    assert out[out['CustomerId']==2]['transaction_count'].iloc[0] == 3
    
def test_time_features_extract():
    df = pd.DataFrame({
        'TransactionStartTime': ['2023-10-01 10:00:00', '2024-01-31 23:59:59']
    })
    tf = TimeFeatures(time_col='TransactionStartTime')
    out = tf.fit_transform(df)
    assert 'transaction_hour' in out.columns
    assert 'transaction_day' in out.columns
    assert (out['transaction_hour'] == [10,23]).all()
    assert out['transaction_month'].iloc[1] == 1

if __name__ == "__main__":
    pytest.main([__file__])
