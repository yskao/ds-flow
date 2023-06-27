import pandas as pd
from mllib.repurchase.hlh_rfm import RFM


def test_hlh_rfm(rfm_data):

    expected_dict = {
        "mobile": "0900104306",
        "frequency": 1,
        "recency": 1279.0,
        "T": 1279.0,
        "max_min_recency": 0,
        "price_sum": 6990.0,
        "price_mean": 6990.0,
        "last_date": "2019-12-18",
        "assess_date": "2023-06-19",
        "repurchase_period_mean": 0,
        "HS-91APP": 1.0,
        "HS-OLD": 0.0,
        "POS": 0.0,
        "quantity_sum": 1.0,
        "quantity_mean": 1.0,
    }
    expected_result = pd.DataFrame(expected_dict, index=[0])

    rfm_result = RFM(
        transaction_df=rfm_data,
        customer_id_col="mobile",
        datetime_col="order_date",
        price_col="sales_amount",
        start_date="2000-01-01",
        assess_date="2023-06-19",
        quantity_col="sales_quantity",
        extra_features=["HS-91APP", "HS-OLD", "POS"],
    ).get_rfm_df()

    assert all(expected_result == rfm_result)


def test_hlh_seasonal_rfm(rfm_data):

    expected_dict = {
        "mobile": "0900104306",
        "seasonal": 4,
        "frequency": 2,
        "recency": 618.0,
        "T": 1279.0,
        "max_min_recency": 0,
        "price_sum": 13980.0,
        "price_mean": 6990.0,
        "last_date": "2019-12-18",
        "assess_date": "2023-06-19",
        "repurchase_period_mean": 661.0,
        "HS-91APP": 2.0,
        "HS-OLD": 0.0,
        "POS": 0.0,
        "quantity_sum": 2.0,
        "quantity_mean": 1.0,
    }
    expected_result = pd.DataFrame(expected_dict, index=[0])

    rfm_result = RFM(
        transaction_df=rfm_data,
        customer_id_col="mobile",
        datetime_col="order_date",
        price_col="sales_amount",
        start_date="2000-01-01",
        assess_date="2023-06-19",
        time_feature="seasonal",
        quantity_col="sales_quantity",
        extra_features=["HS-91APP", "HS-OLD", "POS"],
    ).get_rfm_df()

    assert all(expected_result == rfm_result)
