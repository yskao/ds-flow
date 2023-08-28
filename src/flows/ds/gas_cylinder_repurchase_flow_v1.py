"""
TODO<Sam> gas_cylinder_repurchase_flow_v1.py 修正 gas_cylinder_repurchase_flow_v1
https://app.asana.com/0/1202942986616169/1205265288881501/f.
"""
from pathlib import Path
from typing import TypeVar

import joblib
import numpy as np
import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter
from google.cloud.storage import Client as GCSClient
from mllib.data_engineering import gen_dummies, gen_repurchase_train_and_test_df
from mllib.data_extraction import ExtractDataForTraining
from mllib.ml_utils.cylinder_repurchase_utils import cal_cylinder_purchase_qty
from mllib.ml_utils.utils import model_upload_to_gcs, upload_df_to_bq
from mllib.repurchase.hlh_repurchase import HLHRepurchase
from mllib.sql_query.soda_stream_repurchase_script import (
    cdp_soda_stream_campaign_sql,
    cdp_soda_stream_sql,
)
from prefect import flow, get_run_logger, task

from utils.gcp.client import get_bigquery_client, get_gcs_client
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
    "HS-91APP": [1, 0, 0],
    "HS-OLD": [0, 1, 0],
    "POS": [0, 0, 1],
}

mapping_combo_qty = {
    "2入": 2,
    "3入": 3,
    "4入": 4,
    "5入": 5,
    "二入": 2,
    "三入": 3,
    "四入": 4,
    "五入": 5,
}

fillna_values = {
    "TY_Point_All_Cnt": 0, "TY_Point_UpperLimit_Cnt": 0, "TY_Point_Used_Cnt": 0,
    "TY_Point_Unused_Cnt": 0, "TY_Point_UnusedWOCoupon_Cnt": 0,
    "TY_Coupon_Sent_Cnt": 0, "TY_Coupon_Used_Cnt": 0, "TY_Coupon_Unused_Cnt": 0,
    "LY_Point_All_Cnt": 0, "LY_Point_UpperLimit_Cnt": 0, "LY_Point_Used_Cnt": 0,
    "LY_Point_Unused_Cnt": 0, "LY_Point_UnusedWOCoupon_Cnt": 0,
    "LY_Coupon_Sent_Cnt": 0, "LY_Coupon_Used_Cnt": 0, "LY_Coupon_Unused_Cnt": 0,
    "Member_GasCylindersReward_Point": 0, "GasCylinders_Purchase_Qty": 0,
}

@task(name="create_prediction_table_for_soda_stream")
def create_ds_soda_stream_prediction_table(bigquery_client: BigQueryClient) -> None:
    """Create DS.DS_SodaStream_Prediction."""
    bigquery_client.query(
        """
            CREATE OR REPLACE TABLE DS.DS_SodaStream_Prediction (
                ETL_Datetime DATETIME NOT NULL OPTIONS(description="ETL執行日期"),
                Assess_Date DATE NOT NULL OPTIONS(description="模型執行日期"),
                Member_Mobile STRING NOT NULL OPTIONS(description="會員手機號碼"),
                User_Mask_Mobile STRING OPTIONS(description="顧客遮罩後手機號碼"),
                Member_GasCylindersReward_Point INT64 OPTIONS(description="鋼瓶集點數"),
                GasCylinders_Purchase_Qty INT64 OPTIONS(description="今年鋼瓶購買數量"),
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
                Repurchase_Flag BOOL OPTIONS(description="用戶 N 天回來的標籤"),
                Repurchase_Possibility FLOAT64 OPTIONS(description="區間內購買機率"),
                TY_Campaign_Year_ID STRING OPTIONS(description="今年參加活動年份"),
                TY_Point_All_Cnt INTEGER OPTIONS(description="今年所有得到的點數"),
                TY_Point_UpperLimit_Cnt INTEGER OPTIONS(description="今年優惠券上限點數"),
                TY_Point_Used_Cnt INTEGER OPTIONS(description="今年扣除已使用優惠券的點數"),
                TY_Point_Unused_Cnt INTEGER OPTIONS(description="今年未使用優惠券的點數"),
                TY_Point_UnusedWOCoupon_Cnt INTEGER OPTIONS(description="今年扣除所有優惠券的點數"),
                TY_Coupon_Sent_Cnt INTEGER OPTIONS(description="今年已發送的優惠券張數"),
                TY_Coupon_Used_Cnt INTEGER OPTIONS(description="今年已使用的優惠券張數"),
                TY_Coupon_Unused_Cnt INTEGER OPTIONS(description="今年未使用的優惠券張數"),
                TY_Coupon_Sent_ID STRING OPTIONS(description="今年已發送的優惠券序號"),
                TY_Coupon_Used_ID STRING OPTIONS(description="今年已使用的優惠券序號"),
                TY_Coupon_Unused_ID STRING OPTIONS(description="今年未使用的優惠券序號"),
                TY_Coupon_LastCreated_DT DATETIME OPTIONS(description="今年最新優惠券產出日期"),
                TY_Coupon_Exp_Date DATE OPTIONS(description="今年優惠券失效日"),
                LY_Campaign_Year_ID STRING OPTIONS(description="去年參加活動年份"),
                LY_Point_All_Cnt INTEGER OPTIONS(description="去年所有得到的點數"),
                LY_Point_UpperLimit_Cnt INTEGER OPTIONS(description="去年優惠券上限點數"),
                LY_Point_Used_Cnt INTEGER OPTIONS(description="去年扣除已使用優惠券的點數"),
                LY_Point_Unused_Cnt INTEGER OPTIONS(description="去年未使用優惠券的點數"),
                LY_Point_UnusedWOCoupon_Cnt INTEGER OPTIONS(description="去年扣除所有優惠券的點數"),
                LY_Coupon_Sent_Cnt INTEGER OPTIONS(description="去年已發送的優惠券張數"),
                LY_Coupon_Used_Cnt INTEGER OPTIONS(description="去年已使用的優惠券張數"),
                LY_Coupon_Unused_Cnt INTEGER OPTIONS(description="去年未使用的優惠券張數"),
                LY_Coupon_Sent_ID STRING OPTIONS(description="去年已發送的優惠券序號"),
                LY_Coupon_Used_ID STRING OPTIONS(description="去年已使用的優惠券序號"),
                LY_Coupon_Unused_ID STRING OPTIONS(description="去年未使用的優惠券序號"),
                LY_Coupon_LastCreated_DT DATETIME OPTIONS(description="去年最新優惠券產出日期"),
                LY_Coupon_Exp_Date DATE OPTIONS(description="去年優惠券失效日")
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

    logging.info("gen_dummies for data_source...")
    dummy = gen_dummies(data["data_source"], mapping_dict=source_map)
    orders_df = pd.concat((data, dummy), axis=1)

    logging.info("removing frequency <= 1")
    match_mobile = (
        orders_df
        .assign(order_date=lambda df: df["order_date"].dt.to_period("D"))
        .groupby("mobile", as_index=False)["order_date"].nunique()
        .query("order_date >= 2") # 兩次消費以上才會納入計算
        ["mobile"]
    )
    orders_correct_df = orders_df[orders_df["mobile"].isin(match_mobile)]

    # 計算實際「鋼瓶」購買數量, 解 combo
    cylinder_purchase_qty_df = cal_cylinder_purchase_qty(
        orders_df=orders_correct_df,
        mapping_combo_qty=mapping_combo_qty,
        assess_date=assess_date,
    ).rename(columns={"mobile": "Member_Mobile", "purchase_qty": "GasCylinders_Purchase_Qty"})

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

    # 季節性用戶資料
    train_seasonal_df, pred_seasonal_df = gen_repurchase_train_and_test_df(
        transaction_df=orders_correct_df,
        customer_id_col="mobile",
        datetime_col="order_date",
        price_col="sales_amount",
        start_date="2000-01-01",
        assess_date=assess_date,
        n_days=120,
        quantity_col="sales_quantity",
        time_feature="seasonal",
        extra_features=["HS-91APP", "HS-OLD", "POS"],
    )

    train_df["mobile"] = train_df["mobile"].astype("category")
    pred_df["mobile"] = pred_df["mobile"].astype("category")
    train_seasonal_df["mobile"] = train_seasonal_df["mobile"].astype("category")
    pred_seasonal_df["mobile"] = pred_seasonal_df["mobile"].astype("category")

    return (
        train_df,
        pred_df,
        train_seasonal_df,
        pred_seasonal_df,
        all_cycle_period_df,
        cylinder_purchase_qty_df,
    )


@task(name="without_seasonal_training")
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


@task(name="with_seasonal_training")
def train_seasonal_model(train_seasonal_df: pd.DataFrame) -> Predictor:
    ml_seasonal_model = HLHRepurchase(
        n_days=120,
        method="ml",
    )
    ml_seasonal_model.fit(
        train_seasonal_df.drop(["last_date", "assess_date"], axis=1),
        target="repurchase_120_flag",
    )
    return ml_seasonal_model


@task(name="store_model_to_cloud_storage")
def store_trained_model_to_gcs(
    model: Predictor,
    model_version: str,
    gcs_client: GCSClient,
) -> None:
    local_model_path = f"soda_stream_repurchase_model_{model_version}.json"
    gcs_model_path = "soda-stream-repurchase/" + local_model_path
    joblib.dump(model.repurchase_model.clf_model, local_model_path)
    model_upload_to_gcs(
        local_model_path=local_model_path,
        gcs_model_path=gcs_model_path,
        bucket_name="ml-project-hlh",
        gcs_client=gcs_client,
    )
    remove_path = Path(local_model_path)
    if remove_path.exists():
        remove_path.unlink()


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
def get_extra_soda_stream_sample(bigquery_client: BigQueryClient) -> pd.DataFrame:
    """Get extra sample to be added to prediciton df."""
    query = """SELECT * FROM DS.DS_SodaStream_Prediction_TestList"""
    return bigquery_client.query(query).result().to_dataframe()


@task
def delete_assess_date_duplicate(bigquery_client: BigQueryClient, assess_date: pd.Timestamp) -> None:
    query_parameters = [
            ScalarQueryParameter("Assess_Date", "STRING", str(assess_date.date())),
        ]
    delete_query = """
        DELETE FROM DS.DS_SodaStream_Prediction
        WHERE Assess_Date = @assess_date
    """
    bigquery_client.query(
        delete_query,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).result()


@task(name="gen_cdp_required_data")
def gen_cdp_soda_stream_data_to_bq(bigquery_client: BigQueryClient):
    return bigquery_client.query(cdp_soda_stream_sql()).result()


@task(name="gen_cdp_required_campaign_data")
def gen_cdp_soda_stream_campaign_data_to_bq(bigquery_client: BigQueryClient):
    return bigquery_client.query(cdp_soda_stream_campaign_sql()).result()


@flow(name=generate_flow_name())
def gas_cylinder_repurchase_flow_v1(init: bool = False) -> None:
    """Flow for ds.ds_sodastream_prediction."""
    bigquery_client = get_bigquery_client()
    gcs_client = get_gcs_client()
    get_cylinder_data_model = ExtractDataForTraining()
    member_cylinder_points_df = (
        get_cylinder_data_model.get_cylinder_points_df(bigquery_client)
        .rename(columns={
            "GasCylinder_Point_Cnt": "Member_GasCylindersReward_Point",
            "Phone": "Member_Mobile",
            },
        )
    )
    sodastream_campaign_last2y_df = (
        get_cylinder_data_model.get_mart_ds_sodastream_campaign_last2y_df(bigquery_client)
        .rename(columns={"User_Mobile": "Member_Mobile"})
        .drop(columns=["ETL_DT"])
    )

    if init:
        create_ds_soda_stream_prediction_table(bigquery_client)

    assess_date = pd.Timestamp.now("Asia/Taipei").tz_localize(None)
    seasonal_value = (assess_date + pd.DateOffset(days=1)).quarter

    (train_df, pred_df, train_seasonal_df,
     pred_seasonal_df, all_cycle_period_df,
     cylinder_purchase_qty_df) = prepare_training_data(
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

    # 季節性預測
    ml_seasonal_model = train_seasonal_model(train_seasonal_df)
    pred_seasonal_result = (
        pd.DataFrame(
            {
                "Member_Mobile": pred_seasonal_df["mobile"],
                "seasonal": pred_seasonal_df["seasonal"],
                "Repurchase_Possibility": soda_stream_repurchase_predict(ml_seasonal_model, pred_seasonal_df),
            },
        )
        .loc[lambda df: df["seasonal"] == seasonal_value,
             ["Member_Mobile", "Repurchase_Possibility"]]
        .reset_index(drop=True)
    )

    bq_df = (
        pred_df
        .merge(pred_result, on="mobile", how="left")
        .assign(
            mobile=lambda df: df["mobile"].astype(str),
            last_date=lambda df: pd.to_datetime(df["last_date"], utc=True).dt.tz_convert("Asia/Taipei"),
            assess_date=lambda df: df["assess_date"].dt.date,
            etl_datetime=pd.Timestamp.now("Asia/Taipei"),
        )
        .rename(bq_columns, axis="columns")
    )

    bq_df = (
        pd.concat((bq_df, no_cycle_period_member_df), axis=0)
        .drop_duplicates("Member_Mobile")
        .reset_index(drop=True)
    )
    # 加入季節性的用戶到原本預測的用戶中
    bq_df_has_seasonal_probability = (
        bq_df.loc[bq_df["Member_Mobile"].isin(pred_seasonal_result["Member_Mobile"]), "Repurchase_Possibility"].dropna())

    pred_seasonal_result["Repurchase_Possibility"] = np.where(
        bq_df_has_seasonal_probability >= pred_seasonal_result["Repurchase_Possibility"],
        bq_df_has_seasonal_probability,
        pred_seasonal_result["Repurchase_Possibility"],
    )

    bq_df.loc[bq_df["Member_Mobile"].isin(pred_seasonal_result["Member_Mobile"]), "Repurchase_Possibility"] = (
        pred_seasonal_result["Repurchase_Possibility"])
    bq_df["Repurchase_Flag"] = np.where(bq_df["Repurchase_Possibility"]>=0.5, 1, 0)
    bq_df["ETL_Datetime"] = bq_df["ETL_Datetime"].fillna(method="ffill")

    # 新增集點資料 - 沒有點數的會員補 0
    # 新增過去兩年鋼瓶活動資料(包含優惠券、點數使用)
    # 新增今年鋼瓶購買數量
    bq_df = (
        bq_df
        .merge(member_cylinder_points_df, on="Member_Mobile", how="left")
        .merge(sodastream_campaign_last2y_df, on="Member_Mobile", how="left")
        .merge(cylinder_purchase_qty_df, on="Member_Mobile", how="left")
    ).fillna(fillna_values)
    # prediction data 上傳資料到 bq
    delete_assess_date_duplicate(bigquery_client, assess_date)
    upload_df_to_bq(
        bigquery_client=bigquery_client,
        upload_df=bq_df,
        bq_table="DS.DS_SodaStream_Prediction",
        bq_project="data-warehouse-369301",
    )

    # 通知名單上傳到 BQ - CDP
    gen_cdp_soda_stream_data_to_bq(bigquery_client)
    gen_cdp_soda_stream_campaign_data_to_bq(bigquery_client)

    # save model
    store_trained_model_to_gcs(
        model=ml_model,
        model_version=assess_date.strftime("%Y-%m-%d"),
        gcs_client=gcs_client,
    )

if __name__ == "__main__":
    gas_cylinder_repurchase_flow_v1(False)
