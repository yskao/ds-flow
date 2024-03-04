

import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient

from mllib.sql_query.soda_stream_repurchase_script import CylinderSQL


class ExtractDataForTraining:
    """可以透過 get_cylinder_df() 取得鋼瓶交易資料."""

    def get_cylinder_df(self, bigquery_client: BigQueryClient) -> pd.DataFrame:

        orders_hs_old_sql_query = CylinderSQL.orders_hs_old_sql()
        orders_hs_91app_sql_query = CylinderSQL.orders_hs_91app_sql()
        orders_pos_sql_query = CylinderSQL.orders_pos_sql()
        products_df_sql_query = CylinderSQL.products_df_sql()

        orders_hs_old_df = bigquery_client.query(orders_hs_old_sql_query).result().to_dataframe()
        orders_hs_91app_df = bigquery_client.query(orders_hs_91app_sql_query).result().to_dataframe()
        orders_pos_df = bigquery_client.query(orders_pos_sql_query).result().to_dataframe()
        products_df = bigquery_client.query(products_df_sql_query).result().to_dataframe()

        orders_df = (
            pd.concat([orders_hs_old_df, orders_hs_91app_df, orders_pos_df])
            .merge(
                products_df[["sku", "product_name", "product_category_id_2", "product_category_2", "product_category_1"]],
                on=["sku", "product_name", "product_category_1"], how="left",
            )
            .assign(
                order_date=lambda df: pd.to_datetime(df["order_date"]),
                sales_amount=lambda df: pd.to_numeric(df["sales_amount"]),
                sales_quantity=lambda df: pd.to_numeric(df["sales_quantity"]),
            )
            .loc[lambda df: df["mobile"].apply(lambda x: len(str(x)) == 10)]
            .loc[lambda df: ~df["mobile"].str.contains(r"/|[a-zA-Z]+")]
            .loc[lambda df: df["order_date"].apply(lambda x: type(x) == pd.Timestamp)]
            # SD9001003A (SODASTREAM 回收快扣空鋼瓶) 37010303 (SODASTREAM 二氧化碳回收空鋼瓶).
            .loc[lambda df: ~(df["product_name"].str.contains("回收", na=False) & (df["sku"].isin(["SD9001003A", "37010303"])))]
        )
        return orders_df


    def get_cylinder_points_df(self, bigquery_client: BigQueryClient) -> pd.DataFrame:
        return bigquery_client.query(CylinderSQL.gas_cylinder_points_sql()).result().to_dataframe()


    def get_mart_ds_sodastream_campaign_last2y_df(self, bigquery_client: BigQueryClient) -> pd.DataFrame:
        return bigquery_client.query(CylinderSQL.mart_ds_sodastream_campaign_last2y_sql()).result().to_dataframe()
