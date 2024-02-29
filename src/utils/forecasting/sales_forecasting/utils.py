from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import pandas_flavor as pf
import yaml
from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.bigquery import LoadJobConfig, QueryJobConfig, ScalarQueryParameter

from utils.tools.common import get_agent_forecast_data, get_product_custom_data_p03

Predictor = TypeVar("Predictor")

def shift_data(
    data: pd.Series,
    lag_shift: int=0,
    future_shift: int=0,
) -> pd.DataFrame:
    """將時間序列資料移動指定的步數."""
    error_msg = "shift value must be integer"
    if not isinstance(future_shift, int) or not isinstance(lag_shift, int):
        raise TypeError(error_msg)

    tmp_future = pd.concat(
        [data.to_frame().shift(-shift_i+1).rename(
            columns=lambda col: f"{col}_future{shift_i}", # noqa
        ) for shift_i in range(1, future_shift+1)],
        axis=1,
    )
    tmp_lag = pd.concat(
        [data.to_frame().shift(i).rename(
            columns=lambda col: f"{col}_lag{i}", # noqa
        ) for i in range(1, lag_shift+1)],
        axis=1,
    )
    return pd.concat((tmp_lag, tmp_future), axis=1)


def shift_all_product_id_data(
    dataset: pd.DataFrame,
    date_col: str,
    product_col: str,
    target: str,
    lag_shift: int=0,
    future_shift: int=0,
) -> pd.DataFrame:

    pids = dataset[product_col].unique()

    tmp_dfs = []
    for pid in pids:
        tmp_i = dataset[dataset[product_col] == pid].copy()
        train_tmp = shift_data(
            data=tmp_i[target],
            lag_shift=lag_shift,
            future_shift=future_shift,
        )
        tmp_df = pd.concat((tmp_i[[date_col, product_col]], train_tmp), axis=1)
        tmp_df = tmp_df[sorted(tmp_df.columns)][lag_shift:-future_shift+1].reset_index(drop=True)
        tmp_dfs.append(tmp_df)
    return pd.concat(tmp_dfs, axis=0).reset_index(drop=True)


@pf.register_dataframe_method
def custom_data_for_reference(df_for_custom: pd.DataFrame, _cols: str="error") -> pd.DataFrame:
    """將測試資料表格化並整理成可供比較的格式."""
    if _cols == "error":
        # 只取前六個預測目標
        assign_feature = "sales_errors"
        assign_cols = df_for_custom.columns.drop(["product_id"])
    else:
        assign_feature = "sales"
        assign_cols = df_for_custom.columns.drop(["product_id"])

    melt_table = pd.melt(
        df_for_custom,
        id_vars=["product_id"],
        value_vars=assign_cols,
        var_name="estimate_date",
        value_name=assign_feature,
        ignore_index=False,
    ).reset_index(names="estimate_version")

    for idx, col in enumerate(assign_cols):
        melt_table.loc[lambda df: df["estimate_date"]==col, "estimate_date"] = ( # noqa
            pd.to_datetime(melt_table["estimate_version"]) + pd.DateOffset(months=idx)
        )
    melt_table["estimate_month_gap"] = (
        pd.to_datetime(melt_table["estimate_date"]).dt.to_period("M").astype(int)+1
        -  pd.to_datetime(melt_table["estimate_version"]).dt.to_period("M").astype(int)
    )
    return melt_table.astype({"estimate_date": "datetime64[ns]", "estimate_version": "datetime64[ns]"})


@pf.register_dataframe_method
def _get_model_custom_data_result(df_for_custom: pd.DataFrame, _cols: str="error") -> pd.DataFrame:
    custom_df = df_for_custom.custom_data_for_reference(_cols=_cols).copy()
    return (
        custom_df.query("estimate_version == estimate_version.max()")
        .rename(columns={"product_id": "product_custom_id"})
        .astype({"estimate_date": "datetime64[ns]", "estimate_version": "datetime64[ns]"})
    )


def gen_model_testing_df(
    model: Predictor,
    target_time: str,
    bigquery_client: BigQueryClient,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """模型測試表和參考表. 返回的第一個表是模型預估和業務預估和實際值的表, 第二個表是模型預估和業務預估的比較, 得到的參考表."""
    target_time = pd.to_datetime(target_time)
    start_month = (target_time - pd.DateOffset(months=12)).strftime("%Y-%m-01")
    custom_pred_df = model.pred_df.custom_data_for_reference(_cols="sales").copy()

    # 取得 p03 客製化品號表
    product_custom_data = get_product_custom_data_p03(bigquery_client)

    # 業務預估取當月預估值即可 gap=1
    agent_forecast_data = get_agent_forecast_data(
        start_month=start_month,
        end_month=target_time,
        bigquery_client=bigquery_client,
    ).query("estimate_month_gap==1")

    # test_data 取預測版本最新的一期
    test_data = model.test_df._get_model_custom_data_result(_cols="sales").rename(columns={"sales": "net_sale_qty"})
    pred_data = (
        model._quantile_result(custom_pred_df.rename(columns={"sales": "net_sale_qty_model"}), target="net_sale_qty_model")
        .rename(columns={"product_id": "product_custom_id"})
        .merge(product_custom_data, on=["product_custom_id"], how="left")
    )
    error_data = model.test_df._get_model_custom_data_result(_cols="error").rename(columns={"sales_errors": "mae_model"})
    final_df = (
        test_data.merge(agent_forecast_data[["estimate_date", "product_custom_id", "net_sale_qty_agent"]], on=["estimate_date", "product_custom_id"], how="left")
        .merge(pred_data, on=["estimate_version", "estimate_date", "product_custom_id", "estimate_month_gap"], how="left")
        .merge(error_data, on=["estimate_version", "estimate_date", "product_custom_id", "estimate_month_gap"], how="left")
    )
    # 對每個品號的模型預估做 aggregate 然後和業務預估進行比較, 如果模型的誤差高於業務預估誤差的 1.1 倍, 那麼模型預估為"低參考"
    # 如果模型預估的誤差低於業務預估誤差的 0.9 倍, 則為"高參考", 否則為"可參考"
    final_df["mae_agent"] = np.abs(final_df["net_sale_qty"] - final_df["net_sale_qty_agent"])
    final_df["mae_model"] = np.abs(final_df["mae_model"])
    reference_df = final_df.groupby("product_custom_id", as_index=False, observed=False)[["mae_model","mae_agent"]].mean()
    reference_df["reference"] = np.where(
        reference_df["mae_model"] > reference_df["mae_agent"]*1.1, "低參考",
        np.where(reference_df["mae_model"] < reference_df["mae_agent"]*0.9, "高參考", "可參考"),
    )
    reference_df.merge(final_df[["product_custom_id", "estimate_version", "brand_id"]], on=["product_custom_id"])
    return final_df, pred_data, reference_df.drop(columns=["mae_model", "mae_agent"])


def predict_and_test_data_to_bq(
    df_for_upload: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> str:
    """將測試資料比較結果存到資料庫."""
    estimate_version = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-01")
    df_for_upload.insert(1, "dept_code", f"{department_code}00")
    df_cols = {
        "estimate_version": str,
        "dept_code": str,
        "brand_id": str,
        "product_custom_id": str,
        "product_custom_name": str,
        "estimate_date": "datetime64[ns]",
        "estimate_month_gap": int,
        "net_sale_qty_model": float,
        "less_likely_lb": float,
        "likely_lb": float,
        "likely_ub": float,
        "less_likely_ub": float,
    }
    query_parameters = [
            ScalarQueryParameter("estimate_version", "STRING", estimate_version),
        ]
    delete_query = """
        DELETE FROM `data-warehouse-369301.src_ds.ds_p03_model_testing`
        WHERE estimate_version = @estimate_version
    """
    bigquery_client.query(
        delete_query,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).result()
    job = bigquery_client.load_table_from_dataframe(
        dataframe=df_for_upload[df_cols.keys()].astype(df_cols),
        destination=bq_table,
        project="data-warehouse-369301",
        job_config=LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return job.state


def load_seasonal_product_ids(yml_file: str) -> list[str]:
    """Read yml file of seasonal products."""
    path = Path(yml_file)
    yml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    seasonal_product_ids = yml_data.get("seasonal_product_id", [])
    return seasonal_product_ids


def reference_data_to_bq(
    reference_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> str:
    """將 reference 資料比較結果存到資料庫."""
    estimate_version = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-01")
    reference_df.insert(1, "dept_code", f"{department_code}00")
    query_parameters = [
            ScalarQueryParameter("estimate_version", "STRING", estimate_version),
        ]
    delete_query = """
        DELETE FROM src_ds.ds_p03_model_referenceable
        WHERE estimate_version = @estimate_version
    """
    bigquery_client.query(
        delete_query,
        job_config=QueryJobConfig(query_parameters=query_parameters),
    ).result()
    job = bigquery_client.load_table_from_dataframe(
        dataframe=reference_df,
        destination=bq_table,
        project="data-warehouse-369301",
        job_config=LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return job.state
