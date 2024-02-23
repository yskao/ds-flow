import pandas as pd
import pendulum
from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter


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
