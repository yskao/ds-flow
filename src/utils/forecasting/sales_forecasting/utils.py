import pandas as pd


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
