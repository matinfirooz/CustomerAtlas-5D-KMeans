import pandas as pd

from src.data_pipeline import clean_transactions, engineer_customer_5d, FIVE_D_FEATURES


def test_feature_engineering_produces_exactly_five_kmeans_dimensions():
    df = pd.DataFrame({
        "InvoiceNo": ["1", "1", "2", "3"],
        "StockCode": ["A", "B", "A", "C"],
        "Quantity": [2, 1, 3, 4],
        "InvoiceDate": pd.to_datetime([
            "2011-01-01", "2011-01-01", "2011-01-05", "2011-01-10"
        ]),
        "UnitPrice": [10.0, 5.0, 10.0, 2.0],
        "CustomerID": [100, 100, 100, 200],
        "Country": ["UK", "UK", "UK", "France"],
    })

    clean = clean_transactions(df)
    customer = engineer_customer_5d(clean)

    assert list(customer[FIVE_D_FEATURES].columns) == FIVE_D_FEATURES
    assert len(customer) == 2
    assert customer.loc[100, "Frequency"] == 2
    assert customer.loc[100, "ProductDiversity"] == 2
