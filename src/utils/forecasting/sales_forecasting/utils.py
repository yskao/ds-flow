import pandas as pd
import pandas_flavor as pf


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
def custom_data_for_reference(ready_for_custom_df: pd.DataFrame, _cols: str="error") -> pd.DataFrame:
    """將測試資料表格化並整理成可供比較的格式."""
    if _cols == "error":
        # 只取前六個預測目標
        assign_feature = "sales_errors"
        assign_cols = ready_for_custom_df.columns.drop(["product_id"])
    else:
        assign_feature = "sales"
        assign_cols = ready_for_custom_df.columns.drop(["product_id"])

    melt_table = pd.melt(
        ready_for_custom_df,
        id_vars=["product_id"],
        value_vars=assign_cols,
        var_name="estimate_date",
        value_name=assign_feature,
        ignore_index=False,
    ).reset_index(names="month_version")

    for idx, col in enumerate(assign_cols):
        melt_table.loc[lambda df: df["estimate_date"]==col, "estimate_date"] = ( # noqa
            pd.to_datetime(melt_table["month_version"]) + pd.DateOffset(months=idx)
        )
    melt_table["estimate_month_gap"] = (
        pd.to_datetime(melt_table["estimate_date"]).dt.to_period("M").astype(int)+1
        -  pd.to_datetime(melt_table["month_version"]).dt.to_period("M").astype(int)
    )
    return melt_table
