# %%
import logging
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas import DataFrame


def fill_in_spaces(fill_df: DataFrame, col_name: str) -> DataFrame:
    """因應儲存格合併(資料只留在第一格),補齊欄位資料."""
    for idx, data in fill_df.iterrows():
        if data[col_name]:
            target_value = data[col_name]
        else:
            fill_df.loc[idx, col_name] = target_value
    fill_df = fill_df.reset_index(drop=True)

    return fill_df


def get_forecast_month_range() -> list[str]:
    """取得待預測月份."""
    now = datetime.now(ZoneInfo("Aisa/Taipei"))
    counting_month = now.replace(day=1).date()
    year_start_dt = counting_month.replace(month=1, day=1)
    next_year_mid = year_start_dt.replace(year=counting_month.year + 1, month=6)
    predict_months = pd.date_range(
        start=counting_month, end=next_year_mid, freq="MS",
    ).date
    month_cols_list = [month.strftime("%Y%m") for month in predict_months]
    logging.info("預估月份為: %s", month_cols_list)

    return month_cols_list


def get_first_product_code(df: DataFrame) -> DataFrame:
    """取得單一品號資料."""
    product_codes_series = df["品號"].str.split("/")
    df["單一品號"] = product_codes_series.apply(lambda x: x[0])

    return df


def get_group_code_dict(agent_customer_info_df: DataFrame, agent_name: str) -> dict[str, list[str]]:

    channel_by_agent = agent_customer_info_df.query(f"業務員名稱 == '{agent_name}'")

    group_code_dict = defaultdict(list)
    for _, row in channel_by_agent.iterrows():
        group_code_dict[row["頁籤分類"]].append(row["集團代號"]) # noqa

    return group_code_dict
