import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

class RFM:
    """
    初始化RFM對象,指定交易數據、客戶ID列、日期列、價格列、計算RFM所需的起始日期和結束日期等參數。.

    Args:
    ----
        transaction_df (pd.DataFrame): 交易數據
        customer_id_col (str): 客戶ID列名
        datetime_col (str): 日期列名
        price_col (str): 價格列名
        start_date (str): 計算RFM所需的起始日期
        assess_date (str): 計算RFM所需的結束日期
        quantity_col (Optional[str], optional): 數量列名(可選)。如果未提供此參數,則不會計算數量相關的RFM值。默認為None。
        freq (str, optional): 日期頻率,如"D"代表天,"W"代表周,"M"代表月,"Q"代表季度。默認為"D"。

    Raises:
    ------
        ValueError: 如果提供的數量列名不存在於交易數據中,則會 ValueError 異常。

    """

    def _get_group_by_cols(
        self,
        customer_df: pd.DataFrame,
        group_cols: list[str],
    ) -> pd.Grouper:
        """根據顧客ID對顧客交易記錄進行分組."""
        return customer_df.groupby(group_cols)


    def _add_quantity(self) -> pd.DataFrame:
        """添加顧客購買數量的統計信息."""
        return (
            self.prepared_customer_group
            [self.quantity_col]
            .agg(["sum", "mean"])
            .rename(columns={"sum": "quantity_sum", "mean": "quantity_mean"})
            .reset_index()
        )


    @staticmethod
    def _get_time_tag(transaction_df: pd.DataFrame, datetime_col: str, time_feature: str) -> pd.DataFrame:
        tmp_df = transaction_df.copy()
        if time_feature == "seasonal":
            tmp_df.loc[:, "seasonal"] = transaction_df[datetime_col].dt.quarter.astype("category")
        elif time_feature == "month":
            tmp_df.loc[:, "month"] = transaction_df[datetime_col].dt.month.astype("category")
        else:
            pass
        return tmp_df


    def _prepare_customer_group(self) -> pd.DataFrame:
        """根據指定時間範圍對顧客交易記錄進行預處理."""
        transaction_df_copy = self.transaction_df.copy()
        transaction_df_copy[self.datetime_col] = pd.to_datetime(self.transaction_df[self.datetime_col]).dt.to_period("D")
        transaction_df_copy[self.customer_id_col] = self.transaction_df[self.customer_id_col].astype("category")
        if self.time_feature:
            transaction_df_copy = self._get_time_tag(transaction_df_copy, self.datetime_col, self.time_feature)
            self.group_cols = [self.customer_id_col, self.time_feature]
        else:
            self.group_cols = [self.customer_id_col]
        prepared_customer_group = self._get_group_by_cols(
            customer_df=transaction_df_copy.query(f"'{self.start_date}' <= {self.datetime_col} < '{self.assess_date}'"),
            group_cols=self.group_cols,
        )
        return prepared_customer_group


    def __init__(
        self,
        transaction_df: pd.DataFrame,
        customer_id_col: str,
        datetime_col: str,
        price_col: str,
        start_date: str,
        assess_date: str,
        time_feature: str | None=None,
        extra_features: list[str] | None=None,
        quantity_col: str | None=None,
        freq="D",
    ) -> None:
        """初始化 RFM 模型."""
        # 如果指定了 quantity_col,但 transaction_df 中不存在,則引發錯誤
        error_msg = "quantity column %s cannot be found!" % quantity_col
        if quantity_col is not None and quantity_col not in transaction_df.columns:
            raise ValueError(error_msg)

        # 初始化類別變數
        self.transaction_df=transaction_df
        self.customer_id_col=customer_id_col
        self.datetime_col=datetime_col
        self.price_col=price_col
        self.start_date=start_date
        self.assess_date=assess_date
        self.quantity_col=quantity_col
        self.freq=freq
        self.time_feature=time_feature
        self.extra_features=extra_features


    def get_rfm_df(self) -> pd.DataFrame:
        """取得 RFM 資料."""
        # 計算RFM值
        logging.info("get_rfm_df")
        logging.info("prepared_customer_group")
        self.prepared_customer_group=self._prepare_customer_group()
        max_min_date_df = self.prepared_customer_group[self.datetime_col].agg(["max", "min"])
        logging.info("frequency")
        frequency = self.prepared_customer_group[self.datetime_col].nunique().reset_index(name="frequency")
        logging.info("recency")
        recency = (
            ((pd.to_datetime(self.assess_date).to_period("D") - max_min_date_df) / np.timedelta64(1, self.freq))
            .rename(columns={"max": "recency", "min": "T"})
            .reset_index()
        )
        logging.info("max_min_recency")
        max_min_recency = (
            ((max_min_date_df["max"] - max_min_date_df["min"]) / np.timedelta64(1, self.freq))
            .reset_index(name="max_min_recency")
        )
        logging.info("price")
        price = (
            self.prepared_customer_group[self.price_col]
            .agg(["sum", "mean"])
            .rename(columns={"sum": "price_sum", "mean": "price_mean"})
            .reset_index()
        )
        logging.info("last_date_and_repurchase_period")
        last_date_and_repurchase_period=(
            max_min_date_df["max"].dt.to_timestamp()
            .reset_index(name="last_date")
            .assign(
                assess_date=pd.to_datetime(self.assess_date),
                repurchase_period_mean=(
                    np.nan_to_num(max_min_recency["max_min_recency"] / (frequency["frequency"]-1), nan=0)
                ),
            )
        )
        logging.info("combined_rfm")
        rfm_df = (
            frequency
            .merge(recency, on=self.group_cols)
            .merge(max_min_recency, on=self.group_cols)
            .merge(price, on=self.group_cols)
            .merge(last_date_and_repurchase_period, on=self.group_cols)
        )
        if self.extra_features:
            logging.info("extra_features")
            rfm_df = rfm_df.merge(
                self.prepared_customer_group[self.extra_features].sum(),
                on=self.group_cols,
            )
        if self.quantity_col:
            logging.info("quantity_col")
            rfm_df = rfm_df.merge(
                self._add_quantity(),
                on=self.group_cols,
                how="left",
            )
        logging.info("fillna")
        # fill nan value
        rfm_df_dtypes = rfm_df.dtypes
        rfm_df[rfm_df.select_dtypes(exclude="category").columns] = (
            rfm_df[rfm_df.select_dtypes(exclude="category").columns].fillna(0)
        )
        logging.info("finish")
        return rfm_df.astype(rfm_df_dtypes)
