
from __future__ import annotations

import numpy as np
import pandas as pd
import pendulum
from utilities import common_tools as ct
from utilities.google_sheets import get_google_sheet_client

from mllib.data_engineering import add_product_id_combo_column, prepare_predict_table_to_sql


def predict_data_to_sql(
        predict_df: pd.DataFrame,
        product_info: pd.DataFrame,
        db_table_name: str,
        department_code: str,
) -> None:
    """
    將預測資料存到資料庫.

    Args:
    ----
        predict_df (pd.DataFrame): 預測資料
        product_info (pd.DataFrame): 產品資訊表
        db_table_name (str): 資料庫表名稱
        department_code (str): 部門代碼

    Returns:
    -------
        None
    """
    predict_df_cols = [
        "month_version",
        "dep_code",
        "brand",
        "cat_code_1",
        "cat_code_2",
        "cat_code_3",
        "product_id_combo",
        "product_name",
        "date",
        "predicted_on_date",
        "M",
        "sales_model",
        "less_likely_lb",
        "likely_lb",
        "likely_ub",
        "less_likely_ub",
    ]

    model_version = pendulum.now(tz="Asia/Taipei").strftime("%Y-%m-%d")

    predict_df = prepare_predict_table_to_sql(
        predict_df=predict_df,
        product_data_info=product_info,
        predicted_on_date=model_version,
        department_code=department_code,
    )

    with ct.connect_to_mssql("hlh_psi") as conn:
        predict_df[predict_df_cols].to_sql(
            name=db_table_name,
            con=conn,
            if_exists="append",
            index=False,
        )


def get_product_df(sales_df: pd.DataFrame, training_info: pd.DataFrame) -> pd.DataFrame:
    """
    P02、P03、P04 都能使用的功能,主要將自定品號和銷售資料結合.
    product_id 會轉換成 product_id_combo 方便進行合併.
    """
    product_id_array = training_info["product_id_combo"].unique().astype(str)
    product_df = add_product_id_combo_column(
        sales_df=sales_df,
        product_id_array=product_id_array,
    ).reset_index(drop=True)
    return product_df


def combine_product_target_and_category_for_P03_P04(
    training_info: pd.DataFrame,
    categories_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    合併訓練資料、目標值以及商品類別資訊,並返回結果。.

    參數:
        training_info:pandas DataFrame
            包含商品ID、品牌等商品相關資訊以及目標值的訓練資料。
        categories_df:pandas DataFrame
            包含商品類別資訊的DataFrame。

    返回:
        pandas DataFrame
            合併後的DataFrame,包含商品ID、品牌、商品類別等相關資訊。
    """
    # product_id_combo 有包含 30600041/3060004 這樣個類別
    # explode 可以將該類別資料分離,而我們合併所有資料,只取第一筆 30600041 這筆即可
    product_attr = (
        training_info[["product_id_combo"]]
        .assign(product_id=lambda df: df["product_id_combo"].str.split("/"))
        .explode("product_id")
        .merge(training_info[["product_id_combo", "brand"]], on="product_id_combo")
        .merge(categories_df, on="product_id")
        .assign(
            first_ind=lambda df: df.groupby("product_id_combo")[
                "product_id_combo"
            ].shift(1),
        )
        .loc[lambda df: df["first_ind"].isna(), :]
        .drop(columns=["product_id", "first_ind"])
        .assign(
            cat_code_3=lambda df: np.where(
                df["cat_code_3"] == "無小分類", "", df["cat_code_3"],
            ),
        )
    )
    return product_attr

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


def get_test_data_for_reference(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    將測試資料表格化並整理成可供比較的格式.

    Args:
    ----
        test_df (pd.DataFrame): 測試資料

    Returns:
    -------
        pd.DataFrame: 測試資料的整理表格
    """
    # 只取前六個預測目標
    error_cols = [
        "sales_error1",
        "sales_error2",
        "sales_error3",
        "sales_error4",
        "sales_error5",
        "sales_error6",
    ]

    melt_table = pd.melt(
        test_df,
        id_vars=["product_id_combo"],
        value_vars=error_cols,
        ignore_index=False,
    ).reset_index()

    for idx, col in enumerate(error_cols):
        melt_table.loc[lambda df: df['variable']==col, 'variable'] = ( # noqa
            melt_table["date"] + pd.DateOffset(months=idx+1))

    melt_table = (
        melt_table.rename(
        columns={"date":"predicted_on_date", "variable": "date", "value": "sales_errors"})
        .assign(
            M=lambda df: pd.to_datetime(df["date"]).dt.to_period("M").astype(int)
            - df["predicted_on_date"].dt.to_period("M").astype(int),
            date=lambda df: pd.to_datetime(df["date"]))
    )
    return melt_table


def get_mae_diff(melt_table: pd.DataFrame) -> pd.DataFrame:
    """
    比較模型與業務預估誤差,暫定用 +- 10% 來做可參考範圍.

    Args:
    ----
        melt_table (pd.DataFrame): 測試資料的整理表格

    Returns:
    -------
        pd.DataFrame: 包含比較結果的表格
    """
    calculate_df = melt_table.copy()
    calculate_df["AE_model"] = calculate_df["sales_errors"].abs()
    calculate_df["AE_agent"] = (calculate_df["sales_agent"] - calculate_df["sales"]).abs()
    mae = calculate_df.groupby("product_id_combo")[["AE_model","AE_agent"]].mean().reset_index()

    for percent_diff in [_/10 for _ in range(1,10)]:
        mae['bound_'+str(percent_diff)] = mae.apply(lambda x: {'ub': x['AE_agent']*(1+percent_diff), 'lb': x['AE_agent']*(1-percent_diff)} ,axis=1) # noqa

    for percent_diff in [_/10 for _ in range(1,10)]:
        mae["bound_"+str(percent_diff)+"_flag"] =  (
            mae.apply(lambda x: '低參考' if x['AE_model'] > x['bound_'+str(percent_diff)]['ub']  # noqa
                      else ('高參考' if x['AE_model'] < x['bound_'+str(percent_diff)]['lb'] else '可參考'  # noqa
                            ) ,axis=1)
        )
    return mae



def reference_data_to_sql(mae_df: pd.DataFrame, db_table: str, department_code: str) -> None:
    """
    將測試資料比較結果存到資料庫.

    Args:
    ----
        mae_df (pd.DataFrame): 包含比較結果的表格
        db_table (str): 資料庫表名稱
        department_code (str): 部門代碼

    Returns:
    -------
        None
    """
    model_version = pendulum.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
    mae_df.insert(0, "month_version", model_version)
    mae_df["department_code"] = f"{department_code}00"
    with ct.connect_to_mssql("hlh_psi") as conn:
        mae_df.to_sql(name=db_table, con=conn, if_exists="append", index=False)


def get_time_tag(transaction_df: pd.DataFrame, datetime_col: str, time_feature: str) -> pd.DataFrame:
    transaction_df[datetime_col] = pd.to_datetime(transaction_df[datetime_col])
    if time_feature == "seasonal":
        transaction_df["seasonal"] = transaction_df[datetime_col].dt.quarter.astype("category")
    elif time_feature == "month":
        transaction_df["month"] = transaction_df[datetime_col].dt.month.astype("category")
    else:
        pass
    return transaction_df
