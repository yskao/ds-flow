
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from xgboost import XGBRegressor

from mllib.data_engineering import shift_all_product_id_data


class HLHMLForecast:
    """用於銷售預測功能。使用者需要提供訓練數據,並可以根據需要指定超參數進行訓練和預測."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        n_estimators: int=300,
        future_shift: int=6,
        lag_shift: int=12,
    ) -> None:
        """初始化 HLHMLForecast 物件."""
        error_msg = "lag_shift must be greater than future_shift"
        if lag_shift < future_shift:
            raise ValueError(error_msg)

        error_msg = f"the month must be at least more than {future_shift+lag_shift+1}"
        if len(dataset.groupby("date")["date"]) < (future_shift+lag_shift+1):
            raise ValueError(error_msg)

        self.dataset=dataset
        self.target = target
        self.future_shift = future_shift
        self.lag_shift = lag_shift

        self.model = XGBRegressor(
            tree_method="hist",
            enable_categorical=True,
            random_state=12,
            n_estimators=n_estimators,
        )

        self.train_df = shift_all_product_id_data(
            dataset=dataset,
            target=target,
            product_id_target="product_id_combo",
            lag_shift=self.lag_shift,
            future_shift=self.future_shift,
        ).set_index("date")

        self.cate_cols = ["product_id_combo"]
        self.predict_cols = [f"{target}_future{i}" for i in range(1, future_shift+1)]
        self.train_cols = self.cate_cols + [f"{target}_lag{i}" for i in range(1, lag_shift+1)]

        for col in self.cate_cols:
            self.train_df[col] = self.train_df[col].astype("category")

        self.X = self.train_df[self.train_cols]
        self.y = self.train_df[self.predict_cols]


    def fit(self) -> None:
        """設定 early_stopping_rounds 參數為 2,避免過度擬合."""
        self.model.set_params = {"early_stopping_rounds": 2}
        self.model.fit(self.X, self.y, eval_set=[(self.X, self.y)])
        errors = self.y - self.model.predict(self.X)
        self.errors = errors.rename(
            columns=lambda col: col.replace("future", "error"),
            ).assign(product_id_combo=self.X["product_id_combo"])
        self.test_df = self.y.assign(product_id_combo=self.X["product_id_combo"])
        self.pred_df = (
            pd.DataFrame(self.model.predict(self.X), index=self.y.index, columns=self.predict_cols)
            .assign(product_id_combo=self.X["product_id_combo"])
        )


    def predict(self, x: pd.DataFrame) -> np.array:
        """模型預測結果."""
        return self.model.predict(x)


    def _seasonal_product_forecast(
        self,
        seasonal_product_df: pd.DataFrame,
        forecast_steps: int = 12,
    ) -> pd.DataFrame:
        """針對季節性商品進行時間序列預測."""
        tmp = seasonal_product_df.reset_index(drop=True)[["date", "sales"]].copy()
        time_series_tmp = tmp.set_index("date").squeeze()
        fill_value = time_series_tmp.describe()["25%"]
        time_series_tmp[time_series_tmp < 0] = fill_value
        time_series_tmp.index = pd.DatetimeIndex(time_series_tmp.index, freq="MS")
        stlf = STLForecast(time_series_tmp, ExponentialSmoothing, model_kwargs={"trend": True}, robust=True)
        es_model = stlf.fit()
        es_predict = es_model.forecast(forecast_steps)
        es_predict[es_predict < 0] = fill_value
        return es_predict.reset_index().rename(columns={"index": "date", 0: "sales"})


    def _quantile_result(self, forecast_df: pd.DataFrame, target: str="sales") -> pd.DataFrame:
        """根據預測的差計算區間, 並加入到預測結果 DataFrame 中."""
        return forecast_df.assign(
            less_likely_lb=forecast_df[target]*0.15,
            likely_lb=forecast_df[target]*0.25,
            likely_ub=forecast_df[target]*1.25,
            less_likely_ub=forecast_df[target]*2,
          )


    def rolling_forecast(
        self,
        product_id: list,
        seasonal_product_id: list | None = None,
        n_periods: int = 1,
    ) -> pd.DataFrame:
        """針對給定的產品 ID 預測未來多期銷售量."""
        self.n_periods = n_periods
        forecast_dfs = []
        self.quantile_results = []
        seasonal_product_id = [] if seasonal_product_id is None else seasonal_product_id

        for pid in product_id:
            train_tmp = self.train_df[self.train_df["product_id_combo"] == pid].copy()
            # 針對非季節性商品進行 multi-output 預測
            if pid not in seasonal_product_id:
                tmp_x = np.array(train_tmp[self.X.columns]).tolist()
                tmp_y = np.array(train_tmp[self.y.columns]).tolist()

                for _i in range(n_periods):

                    input_seq = tmp_x[-1].copy()
                    input_seq[self.future_shift+1:] = input_seq[len(self.cate_cols):(self.lag_shift-self.future_shift)+1]
                    input_seq[len(self.cate_cols):self.future_shift+1] = tmp_y[-1][::-1]
                    tmp_x.append(input_seq)

                    input_seq_df = pd.DataFrame([input_seq], columns=self.X.columns).astype({"product_id_combo": "category"})
                    self.input_seq_df = input_seq_df

                    tmp_yi = self.model.predict(self.input_seq_df)[0]
                    tmp_yi = np.where(tmp_yi > np.max(tmp_y), np.max(tmp_y), tmp_yi)
                    tmp_yi = np.where(tmp_yi < np.min(tmp_y), np.min(tmp_y), tmp_yi)
                    tmp_y.append(tmp_yi)

                last_date = train_tmp.index.max() + pd.DateOffset(months=self.future_shift-1)
                dates = pd.date_range(start=last_date, periods=n_periods*self.future_shift+1, freq="MS")
                self.dates = dates[dates != last_date]

                forecast_df = pd.DataFrame(
                    {"date": self.dates},
                ).assign(
                    product_id_combo=pid,
                    month=lambda df: df["date"].dt.month,
                )

                # 利用權重修正預測資料
                tmp_y[-1][0] = tmp_y[-2][-1] * 0.2 + tmp_y[-1][0] * 0.8
                for i in range(1, len(tmp_y[-1])):
                    tmp_y[-1][i] = tmp_y[-1][i-1] * 0.2 + tmp_y[-1][i] * 0.8

                # 負數處理成正數
                tmp_y = np.abs(tmp_y)

                self.forecast_df = pd.concat(
                    (forecast_df, pd.concat(
                        [pd.Series(i, name=self.target) for i in tmp_y[-n_periods:]],
                    ).reset_index(drop=True),
                ), axis=1)

            # 如果是「季節性」商品,利用傳統時間序列滾動進行預測
            else:
                train_tmp = self.dataset[self.dataset["product_id_combo"] == pid].copy()
                ts_train_df = train_tmp.reset_index()[["date", "sales"]]
                ts_train_df.columns = ["date", "sales"]
                forecast_df = self._seasonal_product_forecast(
                    seasonal_product_df=ts_train_df,
                    forecast_steps=n_periods*6, #n_periods*6=12筆
                )
                self.forecast_df = forecast_df.assign(
                    product_id_combo=pid,
                    month=forecast_df["date"].dt.month,
                )

            # quantile result
            self.forecast_df = self._quantile_result(self.forecast_df)

            forecast_dfs.append(self.forecast_df)
            logging.debug("forecast_dataframe: %s", self.forecast_df)

        return pd.concat(forecast_dfs, axis=0)
