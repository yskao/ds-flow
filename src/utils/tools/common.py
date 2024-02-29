import pandas as pd
import pendulum
from google.cloud.bigquery import ArrayQueryParameter, QueryJobConfig, ScalarQueryParameter
from google.cloud.bigquery import Client as BigQueryClient


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
