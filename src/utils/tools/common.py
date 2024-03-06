from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pendulum
from google.cloud.bigquery import (
    ArrayQueryParameter,
    LoadJobConfig,
    QueryJobConfig,
    ScalarQueryParameter,
)
from google.cloud.bigquery import Client as BigQueryClient

from utils.google_sheets import get_google_sheet_client

if TYPE_CHECKING:
    from google.cloud.bigquery import Client as BigQueryClient
    from google.cloud.storage import Client as GCSClient

def get_net_month_qty_p03_data(
    start_date: str,
    end_date: str,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """從指定的日期區間取得三處每月淨銷量數據."""
    sql = """
        SELECT
            Order_YM AS order_ym
            , Product_Custom_ID AS product_custom_id
            , Net_Sale_Qty AS net_sale_qty
        FROM `data-warehouse-369301.dbt_mart_ds.mart_ds_s_net_qty_p03_m_v`
        WHERE Order_YM BETWEEN @start_date AND @next_month_date
    """
    start_date = pendulum.parse(start_date).replace(day=1).to_date_string()
    end_date = pendulum.parse(end_date).replace(day=1).to_date_string()
    query_parameters = [
        ScalarQueryParameter("next_month_date", "STRING", end_date),
        ScalarQueryParameter("start_date", "STRING", start_date),
    ]
    sales_df = bigquery_client.query(
        query=sql,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).to_dataframe().astype({"order_ym": "datetime64[ns]", "net_sale_qty": float})
    return sales_df


def get_agent_forecast_data(
    start_month: str,
    end_month: str,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """取得業務預測資料."""
    month_versions_range = pd.date_range(start=start_month, end=end_month,freq="MS") - pd.DateOffset(months=1)
    month_versions_range_quot_str = [month.strftime("%Y-%m") for month in month_versions_range]
    query_parameters = [
            ArrayQueryParameter("month_versions_range_quot_str", "STRING", month_versions_range_quot_str),
        ]
    sql = """
            WITH source AS (
                SELECT
                    estimate_date
                    , etl_year_month
                    , product_custom_id
                    , sell_in_sale_estimate_net_qty
                    , ROW_NUMBER() OVER (
                        PARTITION BY estimate_date, etl_year_month, sell_in_sale_estimate_net_qty, product_custom_id ORDER BY etl_time DESC, sell_in_sale_estimate_net_qty DESC
                    ) AS rn
                FROM `data-warehouse-369301.ods_HLH_PSI.f_sales_forecast_hist`
                WHERE
                    etl_year_month in UNNEST(@month_versions_range_quot_str)
            )
            , agg_source AS (
                SELECT
                    estimate_date
                    , CAST(CONCAT(etl_year_month, "-01") AS DATE) AS etl_year_month
                    , product_custom_id
                    , SUM(sell_in_sale_estimate_net_qty) AS net_sale_qty_agent
                FROM source
                WHERE rn = 1
                GROUP BY etl_year_month, product_custom_id, estimate_date
                ORDER BY product_custom_id, etl_year_month ASC, estimate_date ASC
            )
            SELECT
                *
                , DATE_DIFF(estimate_date, etl_year_month, MONTH) + 1 AS estimate_month_gap
            FROM agg_source
            WHERE DATE_DIFF(estimate_date, etl_year_month, MONTH) + 1 BETWEEN 1 AND 3 -- 取得業務未來三個月的預測淨銷量
        """
    agent_forecast_df = bigquery_client.query(
        sql,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).to_dataframe(dtypes={"etl_year_month": "datetime64[ns]", "estimate_date": "datetime64[ns]"})
    return agent_forecast_df


def get_product_custom_data_p03(bigquery_client: BigQueryClient) -> pd.DataFrame:
    """p03 自訂產品表."""
    sql= """
        SELECT
            Brand_ID AS brand_id
            , Brand_Custom_Nm AS brand_custom_name
            , Product_Custom_ID AS product_custom_id
            , Product_Custom_Nm AS product_custom_name
        FROM `data-warehouse-369301.dbt_mart_bi.legacy_mart_bi_dim_psi_custom_product_t`
    """
    return bigquery_client.query(sql).to_dataframe()


def get_p02_training_target() -> pd.DataFrame:
    """
    從 Google Sheets 中獲取二處電池的訓練目標值,並返回結果。.

    返回:
        pandas DataFrame
            包含品號和SPU的DataFrame。
    """
    client = get_google_sheet_client()
    url_spu = (
        "https://docs.google.com/spreadsheets/d/1nk7m2UNt1nCUoT3Zi2swFWXxC7CrrHe50azw36f57qY"
    )
    sh = client.open_by_url(url_spu)
    wks_spu = sh.worksheet_by_title("二處電池SPU")
    spu_df = wks_spu.get_as_df()
    return (
        spu_df[["品號", "SPU"]]
        .replace("", np.nan, regex=True)
        .dropna()
        .rename(columns={"品號": "product_id_combo"})
    )


def get_time_tag(transaction_df: pd.DataFrame, datetime_col: str, time_feature: str) -> pd.DataFrame:
    transaction_df[datetime_col] = pd.to_datetime(transaction_df[datetime_col])
    if time_feature == "seasonal":
        transaction_df["seasonal"] = transaction_df[datetime_col].dt.quarter.astype("category")
    elif time_feature == "month":
        transaction_df["month"] = transaction_df[datetime_col].dt.month.astype("category")
    else:
        pass
    return transaction_df


def model_upload_to_gcs(
    local_model_path: str,
    gcs_model_path: str,
    bucket_name: str,
    gcs_client: GCSClient,
) -> None:
    bucket = gcs_client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_model_path)
    return blob.upload_from_filename(local_model_path)


def model_download_from_gcs(
    local_model_path: str,
    gcs_model_path: str,
    bucket_name: str,
    gcs_client: GCSClient,
) -> None:
    bucket = gcs_client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_model_path)
    return blob.download_to_filename(local_model_path)


def upload_df_to_bq(
    bigquery_client: BigQueryClient,
    upload_df: pd.DataFrame,
    bq_table: str,
    bq_project: str,
    write_disposition: str = "WRITE_APPEND",
) -> str:
    """上傳資料到 BQ."""
    job = bigquery_client.load_table_from_dataframe(
        dataframe=upload_df,
        destination=bq_table,
        project=bq_project,
        job_config=LoadJobConfig(write_disposition=write_disposition),
    ).result()
    return job.state


def upload_directory_to_gcs(
    source_dir: str,
    destination_dir: str,
    bucket_name: str,
    gcs_client: GCSClient,
) -> None:
    bucket = gcs_client.bucket(bucket_name)
    directory_as_path_obj = Path(source_dir)
    paths = directory_as_path_obj.rglob("*")
    file_paths = [path for path in paths if path.is_file()]
    relative_paths = [path.relative_to(source_dir) for path in file_paths]
    # Start the upload.
    for file_path, relative_path in zip(file_paths, relative_paths, strict=False):
        gcs_file_path = destination_dir + str(relative_path)
        blob = bucket.blob(gcs_file_path)
        blob.upload_from_filename(str(file_path))
