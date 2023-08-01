
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
        "SD0902003A/SD0902003L",
        "SD9002002A/SD9002002L",
        "SD9002005A/SD9002005L",
        "37010245/S37010245",
        "370501201ZA/S370501201ZA",
    ]
    return seasonal_product_id
