
from typing import TypeVar

import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from mllib.repurchase.hlh_repurchase import HLHRepurchase

from utils.gcp.client import get_bigquery_client

Predictor = TypeVar("Predictor")


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




# def prepare_training_data():



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


def soda_stream_prediction_flow(init: bool = False):
    """Flow for ds.ds_sodastream_prediction."""
    bigquery_client = get_bigquery_client()

    if init:
        create_ds_soda_stream_prediction_table(bigquery_client)



if __name__ == "__main__":
    soda_stream_prediction_flow(True)


    # # interal will use D-1 for training

    #     'HS-91APP': [0, 0, 1],
    #     'HS-OLD': [0, 1, 0],



    #     orders_df
    #     .assign(order_date=lambda df: df['order_date'].dt.to_period('D'))
    #     .groupby('mobile', as_index=False)['order_date'].nunique()
    #     .query("order_date > 1") # frequency > 1



    # # -----------------------------train------------------------------


    # # -----------------------------predict------------------------------
    #         'ml': pd.DataFrame(
    #         ).round(4)[1.0]
