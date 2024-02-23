

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from mllib.repurchase.hlh_rfm import RFM


def add_product_id_combo_column(
    sales_df: pd.DataFrame,
    product_id_array: np.array,
) -> pd.DataFrame:
    """新增 product_id_combo 欄位."""
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
    """根據時間,設定每個樣本的權重."""
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


def prepare_predict_table_to_sql(
    predict_df: pd.DataFrame,
    product_data_info: pd.DataFrame,
    predicted_on_date: str,
    department_code: str,
) -> pd.DataFrame:

    round_columns = ["sales_model", "less_likely_lb", "likely_lb", "likely_ub", "less_likely_ub"]

    predict_df = predict_df.assign(
        month_version=predicted_on_date,
        dep_code=f"{department_code}00",
        predicted_on_date=pd.to_datetime(predicted_on_date, format="%Y-%m-%d"),
        date=lambda df: pd.to_datetime(df["date"], format="%Y-%m-%d"),
        ).rename(columns={"sales": "sales_model"})

    # 計算模型預測日期和未來預測日期的月份差幾個月: 例如模型 8/1 日預測,預測未來 8~12 月,則 M=1,2,3,4,5
    predict_df["M"] = (
        predict_df.apply(lambda row: relativedelta(row["date"], row["predicted_on_date"]), axis=1)
        .apply(lambda x: x.months + x.years*12 + 1)
    )

    predict_df[round_columns] = predict_df[round_columns].round()
    product_unique_info = product_data_info.groupby("product_id_combo", as_index=False).first()
    return predict_df.merge(product_unique_info, on="product_id_combo")


def gen_dummies(feature_series: pd.Series, mapping_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(feature_series.map(mapping_dict).tolist(), columns=list(mapping_dict.keys()))


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
    transaction_df = transaction_df.query(f"'{start_date}' <= {datetime_col} < '{assess_date}'")
    transaction_df = RFM._get_time_tag(transaction_df, datetime_col, time_feature)
    max_date = transaction_df[datetime_col].max() # 找出所有人中最後一次購買時間取最大值作為標準
    cutoff = max_date - pd.to_timedelta(n_days, unit="D") # 最後一次購買時間 - n_days 作為標籤定義

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


def get_continuing_buying_weights(
    orders_correct_df: pd.DataFrame,
    assess_date: pd.Timestamp,
    summer_period: list | None = None,
    past_year: int = 2,
) -> pd.Series:

    weight_mapping = {
        0: 0,
        1: 0,
        2: 0.1,
        3: 0.2,
        4: 0.3,
        5: 0.4,
    }

    orders_df = orders_correct_df.copy()
    orders_df["month"] = orders_df["order_date"].dt.month
    orders_df["year"] = orders_df["order_date"].dt.year

    past_years = (assess_date - pd.DateOffset(years=past_year)).year
    output_df = pd.DataFrame(columns=np.arange(past_years, assess_date.year))

    seasonal_member = (
        orders_df
        .query(f"'{past_years}' <= order_date < '{assess_date.year}' and month in {tuple(summer_period)}")
        .groupby(["mobile", "year"], as_index=False)["month"].count()
        .pivot_table(values="month", columns="year", index="mobile", fill_value=0)
    )

    concat_df = (
        pd.concat((output_df, seasonal_member), axis=0, join="inner")
        .assign(seasonal_weight=lambda df: np.where(df>=1, 1, 0).sum(1))
    ).assign(seasonal_weight=lambda df: df["seasonal_weight"].map(weight_mapping))

    return concat_df
