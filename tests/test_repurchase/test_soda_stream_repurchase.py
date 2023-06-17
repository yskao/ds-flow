import numpy as np
import pandas as pd
from mllib.data_engineering import (
    gen_dummies,
    gen_repurchase_train_and_test_df,
    remove_english_symbol_for_series,
)
from mllib.repurchase.hlh_repurchase import HLHRepurchase

source_map = {
    "HS-91APP": [0, 0, 1],
    "HS-OLD": [0, 1, 0],
    "POS": [1, 0, 0],
}


class TestSodaStreamRepurchase:
    """test soda stream prediction."""

    def test_prepare_training_data(
            self,
            soda_stream_repurchase_data,
            soda_stream_repurchase_data_train_df_test_df,
            soda_stream_repurchase_data_seasonal_train_df_test_df,
    ):
        data = soda_stream_repurchase_data
        expected_train_df, expected_pred_df = soda_stream_repurchase_data_train_df_test_df
        expected_train_seasonal_df, expected_pred_seasonal_df = (
            soda_stream_repurchase_data_seasonal_train_df_test_df)
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

        assert all(train_df == expected_train_df)
        assert all(pred_df == expected_pred_df)
        assert all(train_seasonal_df == expected_train_seasonal_df)
        assert all(pred_seasonal_df == expected_pred_seasonal_df)


    def test_train_and_predict_model(
        self,
        soda_stream_repurchase_data_train_df_test_df,
        soda_stream_repurchase_data_predictions,
    ):
        train_df, pred_df = soda_stream_repurchase_data_train_df_test_df
        ml_model = HLHRepurchase(
            n_days=120,
            method="ml",
        )
        ml_model.fit(
            train_df.drop(["last_date", "assess_date"], axis=1),
            target="repurchase_120_flag",
        )
        assert ml_model is not None

        predictions = pd.DataFrame(
                ml_model.repurchase_predict(pred_df),
                columns=ml_model.repurchase_model.clf_model.classes_,
            ).round(4)[1.0]
        assert np.allclose(predictions, soda_stream_repurchase_data_predictions)
    # def test_bg_model(self):
