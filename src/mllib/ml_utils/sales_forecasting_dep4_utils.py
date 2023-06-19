
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
        .strftime("%Y-%m-%d")
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
    model_version = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-%d")
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
    model_version = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-%d")
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
