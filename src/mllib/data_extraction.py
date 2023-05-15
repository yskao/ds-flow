
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from utilities import common_tools as ct

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


    def get_product_categories(self) -> pd.DataFrame:
        """取得 HLH 資料庫中的產品類資訊。包含品牌、品號、產品資訊。."""
        query_string = """
        SELECT
            品牌 AS brand,
            品號 AS product_id,
            產品大類 AS cat_code_1,
            產品中類 AS cat_code_2,
            產品小類 AS cat_code_3,
            產品性質,
            庫存屬性,
            會計分類
        FROM
            hlh_dw.dbo.dim_products
        """
        with ct.connect_to_mssql("hlh_dw") as conn:
            product_df = pd.read_sql(query_string, con=conn)
        category_df = product_df.loc[
            lambda df: (df["會計分類"] == "商品存貨") & (df["庫存屬性"] != "贈品"),
            ["brand", "product_id", "cat_code_1", "cat_code_2", "cat_code_3"],
        ]
        return category_df

    # TODO: 之後將 psi.utils.sql.get_sales_data 的共用性解決後,再將這個函式移除
    def get_sales_data(
        self, start_date: str, end_date: str, department_code: str,
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
                sales.日期,
                sales.銷貨數量,
                sales.銷貨數量 - sales.退貨數量 AS 銷貨淨數量,
                sales.銷貨贈品數量,
                sales.銷貨金額,
                sales.銷貨金額 - sales.退貨金額 AS 銷貨淨金額,
                sales.品號,
                事業處代號
            FROM
                hlh_dw.dbo.dwd_sales_and_returns_all AS sales
            LEFT JOIN
                hlh_dw.dbo.dim_departments AS dep
            ON
                sales.完整部門代號 = dep.完整部門代號
            WHERE 事業處代號 like %(department_code)s
            AND 日期 < %(next_month_date)s
            AND 日期 >= %(start_date)s
        """
        params = {
            "department_code": f"{department_code}%%",
            "next_month_date": next_month_date.strftime("%Y-%m-%d"),
            "start_date": start_date,
        }
        with ct.connect_to_mssql("hlh") as conn:
            sales_df = pd.read_sql(sql_sales, conn, params=params)


        if department_code == "P02":
            sales_df["銷貨淨數量"] += sales_df["銷貨贈品數量"]  # 二處的銷貨淨數量要加上銷貨贈品數量

        return sales_df.assign(
            date=lambda df: pd.to_datetime(df["日期"]),
            month_begin_date=lambda df: df["date"].dt.strftime("%Y-%m-01"),
        )


    def get_agent_forecast_data(
        self,
        start_month: str,
        end_month: str,
        training_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        取得業務預測資料,包含 M、date、product_id、sales_agent。
        M: 預測時間差
        start_month: 預測開始月份('%Y-%m-%d')
        end_month: 預測結束月份('%Y-%m-%d').
        """
        month_versions_range = pd.date_range(start=start_month, end=end_month,freq="MS") - pd.DateOffset(months=1)
        month_versions_range_quot_str = [month.strftime("%Y-%m") for month in month_versions_range]
        sql = """
                select
                    月份版本, 日期, 自訂品號, 數量
                from
                    f_sales_forecast_versions
                where
                    月份版本 in %(month_versions_range_quot_str)s
            """
        params = {"month_versions_range_quot_str": month_versions_range_quot_str}
        with ct.connect_to_mssql("hlh_psi") as conn:
            forecast_df = pd.read_sql(sql, conn, params=params)
        forecast_target = forecast_df[forecast_df["自訂品號"].isin(training_info["product_id_combo"])].copy()
        forecast_target[["月份版本", "日期"]] = forecast_target[["月份版本", "日期"]].apply(pd.to_datetime)
        forecast_target = forecast_target.groupby(["自訂品號", "月份版本", "日期"]).agg({"數量": "sum"}).reset_index()
        forecast_target["M"] = (
            (forecast_target["日期"].dt.year - forecast_target["月份版本"].dt.year) * 12
            + (forecast_target["日期"].dt.month - forecast_target["月份版本"].dt.month)
        )
        forecast_target = forecast_target.rename(columns={"自訂品號":"product_id_combo", "日期": "date", "數量":"sales_agent"})
        output_df = (
            forecast_target[["M", "date", "product_id_combo", "sales_agent"]]
            .query("1 <= M <= 4").reset_index(drop=True)
        )
        return output_df


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


    def get_cylinder_df(self):

        orders_hs_old_sql_query = CylinderSQL.orders_hs_old_sql()
        orders_hs_91app_sql_query = CylinderSQL.orders_hs_91app_sql()
        orders_pos_sql_query = CylinderSQL.orders_pos_sql()
        products_df_sql_query = CylinderSQL.products_df_sql()

        with ct.connect_to_mssql("hlh") as conn:
            orders_hs_old_df = pd.read_sql(orders_hs_old_sql_query, conn)

        with ct.connect_to_mssql("hlh") as conn:
            orders_hs_91app_df = pd.read_sql(orders_hs_91app_sql_query, conn)

        with ct.connect_to_mssql("hlh") as conn:
            orders_pos_df = pd.read_sql(orders_pos_sql_query, conn)

        with ct.connect_to_mssql("hlh_dw") as conn:
            products_df = pd.read_sql(products_df_sql_query, conn)

        orders_df = (
            pd.concat([orders_hs_old_df, orders_hs_91app_df, orders_pos_df])
            .merge(
                products_df[["品號", "品名", "產品中類代號", "產品中類", "產品大類"]],
                on=["品號", "品名", "產品大類"], how="left",
            )
            .assign(
                日期=lambda df: pd.to_datetime(df["日期"]),
            )
            .loc[lambda df: df["手機"].apply(lambda x: len(x) == 10)]
        )
        return orders_df
