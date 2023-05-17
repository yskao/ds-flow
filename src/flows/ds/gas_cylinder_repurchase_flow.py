
import logging
from typing import TypeVar

import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from mllib.data_engineering import (
    gen_dummies,
    gen_repurchase_train_and_test_df,
    remove_english_symbol_for_series,
)
from mllib.data_extraction import ExtractDataForTraining
from mllib.repurchase.hlh_repurchase import HLHRepurchase

from utils.gcp.client import get_bigquery_client

Predictor = TypeVar("Predictor")

bq_columns = {
    "etl_datetime": "ETL_Datetime",
    "assess_date": "Assess_Date",
    "mobile": "Member_Mobile",
    "HS-91APP": "Chan_HS91App_Day_Cnt",
    "HS-OLD": "Chan_HSOld_Day_Cnt",
    "POS": "Chan_POS_Day_Cnt",
    "frequency": "Frequency_Day_Cnt",
    "recency": "Recency_Day_Cnt",
    "max_min_recency": "Duration_Day_Cnt",
    "T": "Regis_Duration_Day_Cnt",
    "last_date": "LastPurchase_Datetime",
    "price_sum": "NetSaleTax_Amt",
    "price_mean": "NetSaleTax_PerOrd_Amt",
    "repurchase_period_mean": "Avg_Duration_Day_Cnt",
    "quantity_sum": "GasCylinders_Qty",
    "quantity_mean": "GasCylinders_PerOrd_Qty",
    "ml_label": "Repurchase_Flag",
    "ml": "Repurchase_Possibility",
}

source_map = {
    "HS-91APP": [0, 0, 1],
    "HS-OLD": [0, 1, 0],
    "POS": [1, 0, 0],
}



def create_ds_soda_stream_prediction_table(bigquery_client: BigQueryClient) -> None:
    """Create DS.DS_SodaStream_Prediction."""
    bigquery_client.query(
        """
            CREATE OR REPLACE TABLE DS.DS_SodaStream_Prediction (
                ETL_Datetime DATETIME OPTIONS(description="ETL執行日期"),
                Assess_Date DATE OPTIONS(description="模型執行日期"),
                Member_Mobile STRING OPTIONS(description="會員手機號碼"),
                Chan_HS91App_Day_Cnt INT64 OPTIONS(description="累計從線上91APP的購買天數"),
                Chan_HSOld_Day_Cnt INT64 OPTIONS(description="累計從線上舊官網的購買天數"),
                Chan_POS_Day_Cnt INT64 OPTIONS(description="累計從線下實體店的購買次數"),
                Frequency_Day_Cnt INT64 OPTIONS(description="購買頻率天數"),
                Recency_Day_Cnt INT64 OPTIONS(description="最近一次購買到目前的天數"),
                Duration_Day_Cnt INT64 OPTIONS(description="第一次購買到最後一次購買的天數"),
                Regis_Duration_Day_Cnt INT64 OPTIONS(description="註冊後到目前的天數"),
                LastPurchase_Datetime DATETIME OPTIONS(description="最後一次消費日期"),
                NetSaleTax_Amt FLOAT64 OPTIONS(description="累計的消費金額(含稅)"),
                NetSaleTax_PerOrd_Amt FLOAT64 OPTIONS(description="每一次購買平均消費金額(含稅)"),
                Avg_Duration_Day_Cnt FLOAT64 OPTIONS(description="平均購買週期天數"),
                GasCylinders_Qty INT64 OPTIONS(description="累計鋼瓶數量"),
                GasCylinders_PerOrd_Qty FLOAT64 OPTIONS(description="平均一次購買鋼瓶數量"),
                Repurchase_Flag INT64 OPTIONS(description="用戶 N 天回來的標籤"),
                Repurchase_Possibility FLOAT64 OPTIONS(description="區間內購買機率")
            )
            PARTITION BY Assess_Date
        """,
    ).result()


def prepare_training_data(bigquery_client: BigQueryClient) -> pd.DataFrame:
    logging.info("start to prepare training data...")

    assess_date = pd.Timestamp.now("Asia/Taipei")


    logging.info("downloadind_data_from_bq...")
    data_extraction = ExtractDataForTraining()
    data = data_extraction.get_cylinder_df(bigquery_client)

    logging.info("remove_english_symbol_for_series...")
    correct_mobile_index = remove_english_symbol_for_series(data["mobile"]).index
    orders_df = data.loc[correct_mobile_index]

    logging.info("gen_dummies for data_source...")
    dummy = gen_dummies(orders_df["data_source"], mapping_dict=source_map)
    orders_df = pd.concat((orders_df, dummy), axis=1)

    logging.info("removing frequency <= 1")
    match_mobile = (
        orders_df
        .assign(order_date=lambda df: df["order_date"].dt.to_period("D"))
        .groupby("mobile", as_index=False)["order_date"].nunique()
        .query("order_date > 1")
        ["mobile"]
    )
    orders_correct_df = orders_df[orders_df["mobile"].isin(match_mobile)]

    logging.info("gen training and predicting data...")
    train_df, pred_df = gen_repurchase_train_and_test_df(
        transaction_df=orders_correct_df,
        customer_id_col="mobile",
        datetime_col="order_date",
        price_col="sales_amount",
        start_date="2000-01-01",
        assess_date=assess_date,
        n_days=120,
        quantity_col="sales_quantity",
        extra_features=["HS-91APP", "HS-OLD", "POS"],
    )
    return train_df, pred_df


def train_model(train_df: pd.DataFrame) -> Predictor:
    ml_model = HLHRepurchase(
        n_days=120,
        method="ml",
    )
    ml_model.fit(
        train_df.drop(["last_date", "assess_date"], axis=1),
        target="repurchase_flag",
    )
    return ml_model



def eval_metrics(ml_model: Predictor) -> float:
    roc_auc = ml_model.repurchase_model._get_training_evaluation()
    return roc_auc



def get_feature_importance(ml_model: Predictor) -> pd.DataFrame:
    return ml_model.repurchase_model._get_feature_importance()



def soda_stream_repurchase_predict(
        ml_model: Predictor,
        pred_df: pd.DataFrame,
) -> pd.DataFrame:
    return ml_model.predict(pred_df)




def soda_stream_prediction_flow(init: bool = False) -> None:
    """Flow for ds.ds_sodastream_prediction."""
    bigquery_client = get_bigquery_client()

    if init:
        create_ds_soda_stream_prediction_table(bigquery_client)

    train_df, pred_df = prepare_training_data(bigquery_client)
    ml_model = train_model(train_df)
    result_df = soda_stream_repurchase_predict(ml_model, pred_df)

    bq_df = (
        pred_df
        .merge(result_df, on="mobile")
        .assign(
            mobile=lambda df: df["mobile"].astype(str),
            last_date=lambda df: pd.to_datetime(df["last_date"], utc=True).dt.tz_convert("Asia/Taipei"),
            assess_date=lambda df: df["assess_date"].date(),
            etl_datetime=pd.Timestamp.now("Asia/Taipei"),
        )
        .rename(bq_columns, axis="columns")
    )

    bq_df.to_gbq(
        destination_table="DS.DS_SodaStream_Prediction",
        project_id="data-warehouse-369301",
        if_exists="append",
    )


if __name__ == "__main__":
    soda_stream_prediction_flow()
