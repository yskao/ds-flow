
import numpy as np
import pandas as pd
from mllib.hlh_rfm import RFM


def add_product_id_combo_column(
    sales_df: pd.DataFrame,
    product_id_array: np.array,
) -> pd.DataFrame:
    """
    新增「product_id_combo」欄位到銷售資料 DataFrame.

    Args:
    ----
        sales_df (pd.DataFrame): 銷售資料 DataFrame
        product_id_array (np.array): 包含產品 ID 的 NumPy 陣列

    Returns:
    -------
            pd.DataFrame: 包含新增欄位的銷售資料 DataFrame
    """
    start_dt = sales_df["month_begin_date"].min()
    end_dt = sales_df["month_begin_date"].max()
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="MS")
    product_ids = pd.Series(product_id_array).unique()
    output_list = [
        pd.DataFrame(
            {"date": date_range, "product_id_combo": [product_id] * len(date_range)},
        ) for product_id in product_ids
    ]
    output_df = pd.concat(output_list, axis=0)
    return output_df


def set_training_weight(
    output_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    根據時間設定訓練權重,今年的所有資料為 100,其餘按照年度遞減.

    Args:
    ----
        output_df (pd.DataFrame): 包含產品 ID、時間的 DataFrame

    Returns:
    -------
        pd.DataFrame: 包含訓練權重的 DataFrame
    """
    output_df["date"] = pd.to_datetime(output_df["date"])
    years = output_df["date"].dt.year.unique()
    ratio = 100 / len(years)
    year_weight_mapping = {value: (idx+1)*ratio for idx, value in enumerate(years)}
    output_df["year_weight"] = output_df["date"].dt.year.map(year_weight_mapping)
    return output_df

def shift_data(
    data: pd.Series,
    lag_shift: int=0,
    future_shift: int=0,
) -> pd.DataFrame:
    """
    將時間序列資料移動指定的步數.

    Args:
    ----
        data (pd.Series): 時間序列資料
        lag_shift (int): 往前移動的步數,預設為 0
        future_shift (int): 往後移動的步數,預設為 0

    Returns:
    -------
        pd.DataFrame: 移動後的 DataFrame,包含移動前的資料與移動後的資料
    """
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
    product_id_target: str,
    target: str,
    lag_shift: int=0,
    future_shift: int=0,
) -> pd.DataFrame:

    pids = dataset[product_id_target].unique()

    error_msg = "date column must be exist"
    if "date" not in dataset.columns:
        raise ValueError(error_msg)

    tmp_dfs = []
    for pid in pids:
        tmp_i = dataset[dataset[product_id_target] == pid].copy()
        train_tmp = shift_data(
            data=tmp_i[target],
            lag_shift=lag_shift,
            future_shift=future_shift,
        )
        tmp_df = pd.concat((tmp_i[["date", "product_id_combo"]], train_tmp), axis=1)
        tmp_df = tmp_df[sorted(tmp_df.columns)][lag_shift:-future_shift+1].reset_index(drop=True)
        tmp_dfs.append(tmp_df)
    return pd.concat(tmp_dfs, axis=0).reset_index(drop=True)


def prepare_predict_table_to_sql(
    predict_df: pd.DataFrame,
    product_data_info: pd.DataFrame,
    predicted_on_date: str,
    department_code: str,
) -> pd.DataFrame:

    round_columns = ["sales_model", "less_likely_lb", "likely_lb", "likely_ub", "less_likely_ub"]

    predict_df = predict_df.assign(
        dep_code=f"{department_code}00",
        predicted_on_date=pd.to_datetime(predicted_on_date, format="%Y-%m-%d"),
        month_version=lambda df: df["predicted_on_date"],
        date=lambda df: pd.to_datetime(df["date"], format="%Y-%m-%d"),
        M=lambda df: (
            df["date"].dt.month.astype("int")
            - df["predicted_on_date"].dt.month.astype("int") + 1),
        ).rename(columns={"sales": "sales_model"})

    predict_df[round_columns] = predict_df[round_columns].round()
    product_unique_info = product_data_info.groupby("product_id_combo", as_index=False).first()
    return predict_df.merge(product_unique_info, on="product_id_combo")


def gen_dummies(feature_series: pd.Series, mapping_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(feature_series.map(mapping_dict).tolist(), columns=list(mapping_dict.keys()))


def remove_english_symbol_for_series(series: pd.Series) -> pd.Series:
    return series[~series.str.contains(r"/|[a-zA-Z]+")]


def gen_repurchase_train_and_test_df(
    transaction_df: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    price_col: str,
    start_date: str,
    assess_date: str,
    n_days: int,
    quantity_col: str | None=None,
    time_feature: str | None=None,
    extra_features: list[str] | None=None,
) -> pd.DataFrame:

    # 設定判斷有無回購的基準線
    transaction_df.loc[:, datetime_col] = pd.to_datetime(transaction_df[datetime_col])
    transaction_df = RFM._get_time_tag(transaction_df, datetime_col, time_feature)
    max_date = transaction_df[datetime_col].max()
    cutoff = max_date - pd.to_timedelta(n_days, unit="D")

    # 將 in 和 out 資料區別出來,out 資料用來作為是否有回購的依據
    tmp_in_df = transaction_df.loc[lambda df: df[datetime_col] <= cutoff].copy()
    tmp_out_df = transaction_df.loc[lambda df: df[datetime_col] > cutoff].copy()

    # 回購: 會員 id 表
    group_cols = [customer_id_col, time_feature] if time_feature else [customer_id_col]
    repurchase_index_df = (
        tmp_out_df.groupby(group_cols)
        [datetime_col]
        .agg("count")
        .apply(lambda x: 1 if x > 1 else x)
        .reset_index(name=f"repurchase_{n_days}_flag")
    )

    # 訓練用的 RFM
    rfm_table_for_train = RFM(
        transaction_df=tmp_in_df,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        price_col=price_col,
        start_date=start_date,
        assess_date=assess_date,
        quantity_col=quantity_col,
        time_feature=time_feature,
        extra_features=extra_features,
    ).get_rfm_df()

    # 測試用的 RFM
    rfm_table_for_test = RFM(
        transaction_df=transaction_df,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        price_col=price_col,
        start_date=start_date,
        assess_date=assess_date,
        quantity_col=quantity_col,
        time_feature=time_feature,
        extra_features=extra_features,
    ).get_rfm_df()

    return (
        rfm_table_for_train.merge(repurchase_index_df, on=group_cols, how="left").assign(
            **{customer_id_col: lambda df: df[customer_id_col].astype("category"),
            f"repurchase_{n_days}_flag": lambda df: df[f"repurchase_{n_days}_flag"].fillna(0)},
        ), rfm_table_for_test,
    )
