
from typing import TypeVar

import numpy as np
import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.bigquery import LoadJobConfig
from mllib.data_engineering import (
    gen_dummies,
    gen_repurchase_train_and_test_df,
    remove_english_symbol_for_series,
)
from mllib.data_extraction import ExtractDataForTraining
from mllib.repurchase.hlh_repurchase import HLHRepurchase
from prefect import flow, get_run_logger, task

from utils.gcp.client import get_bigquery_client
from utils.prefect import generate_flow_name

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


@task
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


@task(name="log_data_processing")
def prepare_training_data(bigquery_client: BigQueryClient, assess_date: pd.Timestamp) -> pd.DataFrame:
    logging = get_run_logger()
    logging.info("start to prepare training data...")

    logging.info("downloadind_data_from_bq...")
    data_extraction = ExtractDataForTraining()
    # 取資料時,會包含到目前 BQ 裡面最新的資料
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
        .query("order_date > 2")
        ["mobile"]
    )
    orders_correct_df = orders_df[orders_df["mobile"].isin(match_mobile)]

    logging.info("gen training and predicting data...")
    # 切資料時,會根據 assess_date 定義資料的使用區間
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

    _, all_cycle_period_df = gen_repurchase_train_and_test_df(
        transaction_df=orders_df,
        customer_id_col="mobile",
        datetime_col="order_date",
        price_col="sales_amount",
        start_date="2000-01-01",
        assess_date=assess_date,
        n_days=120,
        quantity_col="sales_quantity",
        extra_features=["HS-91APP", "HS-OLD", "POS"],
    )

    train_df["mobile"] = train_df["mobile"].astype("category")
    pred_df["mobile"] = pred_df["mobile"].astype("category")

    return train_df, pred_df, all_cycle_period_df


@task
def train_model(train_df: pd.DataFrame) -> Predictor:
    ml_model = HLHRepurchase(
        n_days=120,
        method="ml",
    )
    ml_model.fit(
        train_df.drop(["last_date", "assess_date"], axis=1),
        target="repurchase_120_flag",
    )
    return ml_model


@task
def eval_metrics(ml_model: Predictor) -> float:
    roc_auc = ml_model.repurchase_model._get_training_evaluation()
    return roc_auc


@task
def get_feature_importance(ml_model: Predictor) -> pd.DataFrame:
    return ml_model.repurchase_model._get_feature_importance()


@task
def soda_stream_repurchase_predict(
        ml_model: Predictor,
        pred_df: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        ml_model.repurchase_predict(pred_df),
        columns=ml_model.repurchase_model.clf_model.classes_,
    ).round(4)[1.0]


@task
def upload_df_to_bq(bigquery_client: BigQueryClient, upload_df: pd.DataFrame) -> str:
    """上傳資料到 BQ."""
    job = bigquery_client.load_table_from_dataframe(
        dataframe=upload_df,
        destination="DS.DS_SodaStream_Prediction",
        project="data-warehouse-369301",
        job_config=LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return job.state


@flow(name=generate_flow_name())
def soda_stream_prediction_flow(init: bool = False) -> None:
    """Flow for ds.ds_sodastream_prediction."""
    bigquery_client = get_bigquery_client()

    if init:
        create_ds_soda_stream_prediction_table(bigquery_client)

    assess_date = pd.Timestamp.now("Asia/Taipei").tz_localize(None)
    assess_date = "2023-04-01"

    train_df, pred_df, all_cycle_period_df = prepare_training_data(
        bigquery_client=bigquery_client,
        assess_date=assess_date,
    )

    ml_model = train_model(train_df)
    pred_result = pd.DataFrame(
        {
            "mobile": pred_df["mobile"],
            "ml": soda_stream_repurchase_predict(ml_model, pred_df),
        },
    )
    no_cycle_period_member_df = (
        all_cycle_period_df[all_cycle_period_df["frequency"] == 1]
        .rename(bq_columns, axis="columns")
    ).assign(
        LastPurchase_Datetime=lambda df: pd.to_datetime(
            df["LastPurchase_Datetime"], utc=True).dt.tz_convert("Asia/Taipei"),
        Assess_Date=lambda df: pd.to_datetime(df["Assess_Date"], format="mixed").dt.date,
    )

    bq_df = (
        pred_df
        .merge(pred_result, on="mobile", how="left")
        .assign(
            mobile=lambda df: df["mobile"].astype(str),
            last_date=lambda df: pd.to_datetime(df["last_date"], utc=True).dt.tz_convert("Asia/Taipei"),
            assess_date=lambda df: df["assess_date"].dt.date,
            etl_datetime=pd.Timestamp.now("Asia/Taipei"),
            ml_label=lambda df: np.where(df["ml"]>=0.5, 1, 0),
        )
        .rename(bq_columns, axis="columns")
    )

    bq_df = pd.concat((bq_df, no_cycle_period_member_df), axis=0).reset_index(drop=True)
    bq_df["ETL_Datetime"] = bq_df["ETL_Datetime"].fillna(method="ffill")
    upload_df_to_bq(bigquery_client, bq_df)


if __name__ == "__main__":
    soda_stream_prediction_flow(False)
