
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from google.cloud.bigquery import LoadJobConfig

from utils.google_sheets import get_google_sheet_client

if TYPE_CHECKING:
    from google.cloud.bigquery import Client as BigQueryClient
    from google.cloud.storage import Client as GCSClient


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
