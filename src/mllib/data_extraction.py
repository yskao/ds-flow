
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

from mllib.sql_script import CylinderSQL


class ExtractDataForTraining:
    """
    從 HLH 資料庫取得所需資料,為訓練模型做準備。
    可以透過 get_training_target() 取得訓練目標產品資訊。
    可以透過 get_product_categories() 取得產品分類資訊。
    可以透過 get_sales_data() 取得銷售資料。
    可以透過 get_agent_forecast_data() 取得業務預測資料。
    可以透過 get_cylinder_df() 取得鋼瓶交易資料。.
    """

    def get_training_target(self) -> None:
        error_msg = "abstract function"
        raise NotImplementedError(error_msg)


    def get_product_categories(self, bigquery_client: BigQueryClient) -> pd.DataFrame:
        """取得 HLH 資料庫中的產品類資訊。包含品牌、品號 (sku)、產品資訊."""
        query_string = """
        SELECT
            brand,
            sku,
            product_category_1,
            product_category_2,
            product_category_3,
            product_type,
            stock_type,
            accounting_type
        FROM
            dim.products
        """
        product_df = bigquery_client.query(query_string).result()
        category_df = product_df.loc[
            lambda df: (df["accounting_type"] == "商品存貨") & (df["stock_type"] != "贈品"),
            ["brand", "sku", "product_category_1", "product_category_2", "product_category_3"],
        ]
        return category_df


    def get_sales_data(
        self,
        bigquery_client: BigQueryClient,
        start_date: str,
        end_date: str,
        department_code: str,
    ) -> pd.DataFrame:
        """
        取得銷售資料,包含日期、銷貨數量、銷貨淨數量、銷貨贈品數量、銷貨金額、銷貨淨金額、品號、事業處代號。
        start_date: 起始時間
        end_date: 結束時間
        department_code: 處代號 (例如 'P03').
        """
        end_date = pd.to_datetime(end_date)
        next_month_date = datetime(end_date.year, end_date.month, 1)
        sql_sales = """
            SELECT
                sales.order_date,
                sales.order_date,
                sales.sales_quantity,
                sales.sales_quantity - sales.return_quantity AS net_sales_quantity,
                sales.sales_gift_quantity,
                sales.sales_amount,
                sales.sales_amount - sales.return_amount AS net_sales_amount,
                sales.sku,
                dep.division_id
            FROM
                dwd.sales_and_returns_all as sales
            LEFT JOIN
                dim.departments AS dep
            ON
                sales.full_department_id = dep.full_department_id
            WHERE division_id like @department_code
                AND order_date < @next_month_date
                AND order_date >= @start_date
        """
        query_parameters = [
            ScalarQueryParameter("department_code", "STRING", f"{department_code}%%"),
            ScalarQueryParameter("next_month_date", "STRING", next_month_date.strftime("%Y-%m-%d")),
            ScalarQueryParameter("start_date", "STRING", start_date),
        ]
        sales_df = bigquery_client.query(
            query=sql_sales,
            job_config=QueryJobConfig(
                query_parameters=query_parameters,
            ),
        ).result().to_dataframe()

        if department_code == "P02":
            sales_df["net_sales_quantity"] += sales_df["sales_gift_quantity"]  # 二處的銷貨淨數量要加上銷貨贈品數量

        return sales_df.assign(
            date=lambda df: pd.to_datetime(df["order_date"]),
            month_begin_date=lambda df: df["date"].dt.strftime("%Y-%m-01"),
        )


    # def get_agent_forecast_data(
    #     self,
    #     start_month: str,
    #     end_month: str,
    #     training_info: pd.DataFrame,
    # ) -> pd.DataFrame:
    #     """取得業務預測資料."""
    #     sql = """
    #             select
    #                 month_version, date, product_id_combo, sales_quantity
    #             from
    #                 f_sales_forecast_versions
    #             where
    #                 月份版本 in %(month_versions_range_quot_str)s
    #         """
    #     with ct.connect_to_mssql("hlh_psi") as conn:


    #     forecast_target["M"] = (
    #         .query("1 <= M <= 4").reset_index(drop=True)


    def get_output_df(self) -> None:
        error_msg = "abstract function"
        raise NotImplementedError(error_msg)


    def get_predict_data(
        self,
        train_df: pd.DataFrame,
        n_month: int=6,
    ) -> pd.DataFrame:
        """
        插入未來 N 個月待預測的列資料 (預設為6個月)。.

        參數:
        - train_df (pd.DataFrame):訓練用資料表格。
        - n_month (int):欲預測的月份數,預設為 6 個月。

        回傳:
        - predict_df (pd.DataFrame):加入預測資料後的資料表格。
        """
        train_df["date"] = pd.to_datetime(train_df["date"])
        training_max_date = train_df["date"].max()
        predict_start_month = train_df["date"].min()
        predict_end_month = training_max_date + relativedelta(months=n_month)

        date_range = pd.date_range(start=predict_start_month, end=predict_end_month, freq="MS")
        product_ids = train_df["product_id_combo"].unique()
        predict_list = [
            pd.DataFrame(pd.DataFrame(
                {"date":date_range, "product_id_combo":[product_id]*len(date_range)}),
            ) for product_id in product_ids
        ]
        predict_df = pd.concat(predict_list, axis=0)
        product_attr = train_df[["product_id_combo", "brand", "cat_code_1", "cat_code_2", "cat_code_3"]].drop_duplicates()
        target = train_df[["date", "product_id_combo", "sales"]].assign(date=lambda df: pd.to_datetime(df["date"]))
        predict_df = (
            predict_df.merge(product_attr, how="left", on="product_id_combo")
            .merge(target, how="left", on=["date", "product_id_combo"])
        )
        return predict_df


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
            .loc[lambda df: df["order_date"].apply(lambda x: type(x) == pd.Timestamp)]
        )
        return orders_df


    def get_cylinder_points_df(self, bigquery_client: BigQueryClient) -> pd.DataFrame:
        return bigquery_client.query(CylinderSQL.gas_cylinder_points_sql()).result().to_dataframe()
