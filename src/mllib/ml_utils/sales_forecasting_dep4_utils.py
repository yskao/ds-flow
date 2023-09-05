
from __future__ import annotations

import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.bigquery import LoadJobConfig, QueryJobConfig, ScalarQueryParameter

from mllib.data_engineering import prepare_predict_table_to_sql


def predict_data_to_bq(
        predict_df: pd.DataFrame,
        product_info: pd.DataFrame,
        bq_table: str,
        department_code: str,
        bigquery_client: BigQueryClient,
) -> str:
    """將預測資料存到資料庫."""
    predict_df_cols = [
        "month_version",
        "dep_code",
        "brand",
        "product_category_1",
        "product_category_2",
        "product_category_3",
        "product_id_combo",
        "product_name",
        "date",
        "predicted_on_date",
        "M",
        "sales_model",
        "less_likely_lb",
        "likely_lb",
        "likely_ub",
        "less_likely_ub",
    ]
    model_version = (
        pd.Timestamp.now("Asia/Taipei")
        .strftime("%Y-%m-01")
    )
    predict_df = prepare_predict_table_to_sql(
        predict_df=predict_df,
        product_data_info=product_info,
        predicted_on_date=model_version,
        department_code=department_code,
    )
    query_parameters = [
            ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    delete_query = """
        DELETE FROM DS.ds_p04_model_predict
        WHERE month_version = @model_version
    """
    bigquery_client.query(
        delete_query,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).result()
    job = bigquery_client.load_table_from_dataframe(
        dataframe=predict_df[predict_df_cols],
        destination=bq_table,
        project="data-warehouse-369301",
        job_config=LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return job.state


def test_data_to_bq(
    test_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> str:
    """將測試資料比較結果存到資料庫."""
    model_version = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-01")
    test_df.insert(0, "month_version", model_version)
    test_df.insert(1, "dep_code", f"{department_code}00")
    test_df_cols = [
        "month_version",
        "dep_code",
        "brand",
        "product_category_1",
        "product_category_2",
        "product_category_3",
        "product_id_combo",
        "product_name",
        "date",
        "predicted_on_date",
        "M",
        "sales",
        "sales_model",
        "sales_agent",
        "less_likely_lb",
        "likely_lb",
        "likely_ub",
        "less_likely_ub",
        "positive_ind_likely",
        "positive_ind_less_likely",
    ]
    query_parameters = [
            ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    delete_query = """
        DELETE FROM DS.ds_p04_model_testing
        WHERE month_version = @model_version
    """
    bigquery_client.query(
        delete_query,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).result()
    job = bigquery_client.load_table_from_dataframe(
        dataframe=test_df[test_df_cols],
        destination=bq_table,
        project="data-warehouse-369301",
        job_config=LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return job.state


def reference_data_to_bq(
    mae_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> str:
    """將 reference 資料比較結果存到資料庫."""
    model_version = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-01")
    mae_df.insert(0, "month_version", model_version)
    mae_df["dep_code"] = f"{department_code}00"
    query_parameters = [
            ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    delete_query = """
        DELETE FROM DS.ds_p04_model_referenceable
        WHERE month_version = @model_version
    """
    bigquery_client.query(
        delete_query,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).result()
    job = bigquery_client.load_table_from_dataframe(
        dataframe=mae_df,
        destination=bq_table,
        project="data-warehouse-369301",
        job_config=LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return job.state


def dep04_seasonal_product_list() -> None:
    seasonal_product_id = [
        "2610107P",
        "2610109P",
        "2610109BL",
        "2610109RP",
        "2660104BLBT",
        "2660106RG",
        "2660106BKBT",
        "26601901",
        "26601902",
        "2660402/OS0301001N",
        "OS9001001A",
        "OS9001002A",
        "OS0103001A",
        "37010126RD/S37010126RD",
        "37010126WT/S37010126WT",
        "37010127/S37010127",
        "SD0109002A/SD0109002L",
        "37010301/S37010301",
        "37010302/S37010302",
        "37010401/S37010401",
        "37010406/S37010406",
        "37010418/S37010418",
        "37010419/S37010419",
        "37010427CP/S37010427CP",
        "SD0902001A/SD0902001L",
        "SD0902003A/SD0902003L",
        "SD0902002A/SD0902002L",
        "SD9002002A/SD9002002L",
        "SD9002005A/SD9002005L",
        "37010245/S37010245",
        "370501201ZA/S370501201ZA",
        "37010106/S37010106",
        "37010117/S37010117",
        "37010119BK/S37010119BK",
        "37010119RD/S37010119RD",
        "37010119WT/S37010119WT",
        "37010122/S37010122",
        "37010122BK/S37101022BK",
        "370101261/S370101261",
        "370101262/S370101262",
        "37010126BK/S37010126BK",
        "37010126BL/S37010126BL/SD0104008AP02",
        "37010126GN/S37010126GN",
        "37010126GY/S37010126GY",
        "37010126NB/S37010126NB",
        "37010126PH/S37010126PH",
        "37010126PK/S37010126PK",
        "37010126WTC/SD0105001AP01/37010126WTC1",
        "37010127W/S37010127W",
        "SD0109001A/SD0109001L",
        "SD0109003A/SD0109003L",
        "SD0109004A/SD0109004L",
        "SD0109005A/SD0109005L",
        "SD0109006A/SD0109006L",
        "SD0109007A/SD0109007L",
        "SD0109008A/SD0109008L",
        "SD0109009A/SD0109009L",
        "SD0110003A/SD0110003L",
        "SD0110002A/SD0110002L",
        "SD0110001A/SD011001L",
        "SD0112001A/SD0112001L",
        "37010126LINE",
        "37010413/S37010413",
        "370104181",
        "37010425/S37010425",
        "370104251",
        "37010426/S37010426",
        "37010427/S37010427",
        "37010427C/S37010427C",
        "37010427CB/S37010427CB",
        "37010428/S37010428",
        "37010433/S37010433",
        "SD0902004A/SD0902004L",
        "SD9002001A/SD9002001L",
        "SD9002003A/SD9002003L",
        "SD9002004A/SD9002004L",
        "37010246/S37010246",
        "SD9003001A/SD9003001L",
        "SD9003003A/SD9003003L",
        "SD9003008A/SD9003008L",
        "SD9003009A/SD9003009L",
        "SD9003010A/SD9003010L",
        "SD9003011A/SD9003011L",
        "37010247",
        "37010248",
        "37010419LINE",
        "SD9003002A",
        "SD9003004A",
        "SD9001001A/SD9001001L",
        "SD9001002A/SD9001002L",
    ]
    return seasonal_product_id
