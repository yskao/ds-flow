import pandas as pd
from mllib.data_engineering import (
    gen_dummies,
    gen_repurchase_train_and_test_df,
    remove_english_symbol_for_series,
)

source_map = {
    "HS-91APP": [0, 0, 1],
    "HS-OLD": [0, 1, 0],
    "POS": [1, 0, 0],
}


class TestSodaStreamRepurchase:
    """test soda stream prediction."""

    def test_prepare_training_data(self, soda_stream_repurchase_data):
        data = soda_stream_repurchase_data
        correct_mobile_index = remove_english_symbol_for_series(data["mobile"]).index
        orders_df = data.loc[correct_mobile_index]
        dummy = gen_dummies(orders_df["data_source"], mapping_dict=source_map)
        orders_df = pd.concat((orders_df, dummy), axis=1)

        match_mobile = (
            orders_df
            .assign(order_date=lambda df: df["order_date"].dt.to_period("D"))
            .groupby("mobile", as_index=False)["order_date"].nunique()
            .query("order_date > 2")
            ["mobile"]
        )
        orders_correct_df = orders_df[orders_df["mobile"].isin(match_mobile)]
        train_df, pred_df = gen_repurchase_train_and_test_df(
            transaction_df=orders_correct_df,
            customer_id_col="mobile",
            datetime_col="order_date",
            price_col="sales_amount",
            start_date="2023-01-01",
            assess_date="2023-06-01",
            n_days=120,
            quantity_col="sales_quantity",
            extra_features=["HS-91APP", "HS-OLD", "POS"],
        )
        # 季節性用戶資料
        train_seasonal_df, pred_seasonal_df = gen_repurchase_train_and_test_df(
            transaction_df=orders_correct_df,
            customer_id_col="mobile",
            datetime_col="order_date",
            price_col="sales_amount",
            start_date="2023-01-01",
            assess_date="2023-06-01",
            n_days=120,
            quantity_col="sales_quantity",
            time_feature="seasonal",
            extra_features=["HS-91APP", "HS-OLD", "POS"],
        )
        train_df["mobile"] = train_df["mobile"].astype("category")
        pred_df["mobile"] = pred_df["mobile"].astype("category")
        train_seasonal_df["mobile"] = train_seasonal_df["mobile"].astype("category")
        pred_seasonal_df["mobile"] = pred_seasonal_df["mobile"].astype("category")
        train_df.to_csv("/Users/samkao/sam/ds-flow/tests/test_data/test_soda_stream_repurchase/train_df.csv", index=False)
        pred_df.to_csv("/Users/samkao/sam/ds-flow/tests/test_data/test_soda_stream_repurchase/pred_df.csv", index=False)
        train_seasonal_df.to_csv("/Users/samkao/sam/ds-flow/tests/test_data/test_soda_stream_repurchase/train_seasonal_df.csv", index=False)
        pred_seasonal_df.to_csv("/Users/samkao/sam/ds-flow/tests/test_data/test_soda_stream_repurchase/pred_seasonal_df.csv", index=False)
