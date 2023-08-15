import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from mllib.data_extraction import ExtractDataForTraining
from mllib.ml_utils.utils import upload_df_to_bq
from prefect import flow, get_run_logger, task

from utils.gcp.client import get_bigquery_client
from utils.prefect import generate_flow_name


@task(name="create_raw_data_table_for_soda_stream")
def create_raw_data_table_for_soda_stream(bigquery_client: BigQueryClient) -> None:
    """Create DS.DS_SodaStreamRawData."""
    bigquery_client.query(
        """
            CREATE OR REPLACE TABLE DS.DS_SodaStreamRawData (
                etl_time STRING NOT NULL OPTIONS(description="etl執行日期"),
                data_source STRING NOT NULL OPTIONS(description="資料來源"),
                order_date STRING NOT NULL OPTIONS(description="訂單日期"),
                sku STRING NOT NULL OPTIONS(description="品號"),
                sales_quantity INT64 OPTIONS(description="數量"),
                sales_amount FLOAT64 OPTIONS(description="金額"),
                external_order_number STRING OPTIONS(description="外部單號"),
                product_category_1 STRING OPTIONS(description="產品大類"),
                mobile STRING OPTIONS(description="手機號碼"),
                product_name STRING OPTIONS(description="品名"),
                remark STRING OPTIONS(description="備註"),
                product_category_id_2 STRING OPTIONS(description="產品中類代號"),
                product_category_2 STRING OPTIONS(description="產品中類")
            )
        """,
    ).result()


@flow(name=generate_flow_name())
def gas_cylinder_repurchase_raw_data_flow(init: bool = False) -> None:
    logging = get_run_logger()
    etl_datetime = (
         pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).strftime("%Y-%m-%d"))
    bigquery_client = get_bigquery_client()
    data_extraction = ExtractDataForTraining()

    if init:
        logging.info("create raw_data_table")
        create_raw_data_table_for_soda_stream(bigquery_client)

    # 取資料時,會包含到目前 BQ 裡面最新的資料
    logging.info("gen raw_data")
    data = data_extraction.get_cylinder_df(bigquery_client)
    data["sales_quantity"] = data["sales_quantity"].astype(int)
    data.insert(0, "etl_datetime", etl_datetime)

    logging.info("upload raw_data")
    upload_df_to_bq(
        bigquery_client=bigquery_client,
        upload_df=data,
        bq_table="DS.DS_SodaStreamRawData",
        bq_project="data-warehouse-369301",
        write_disposition="WRITE_TRUNCATE",
    )


if __name__ == "__main__":
    gas_cylinder_repurchase_raw_data_flow(False)
