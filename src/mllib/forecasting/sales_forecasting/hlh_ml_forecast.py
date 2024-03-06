
import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from xgboost import XGBRegressor

from utils.forecasting.sales_forecasting.utils import shift_all_product_id_data

EXPERIMENT_NAME = "Sales-Forecasting-P03"
LOG_MODEL_NAME = "sales_forecasting_regressor"

class HLHMLForecast:
    """用於銷售預測功能。使用者需要提供訓練數據,並可以根據需要指定超參數進行訓練和預測."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        date_col: str,
        product_col: str,
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
        if len(dataset.groupby(date_col)[date_col]) < (future_shift+lag_shift+1):
            raise ValueError(error_msg)

        self.dataset=dataset.copy()
        self.target = target
        self.future_shift = future_shift
        self.lag_shift = lag_shift
        self.date_col = date_col
        self.product_col = product_col
        self.parameters = {
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 12,
            "n_estimators": n_estimators,
            "early_stopping_rounds": 20,
            "multi_strategy": "multi_output_tree",
            "eval_metric": ["mae", "rmse"],
        }

        self.train_df = shift_all_product_id_data(
            dataset=dataset,
            target=target,
            date_col=date_col,
            product_col=product_col,
            lag_shift=self.lag_shift,
            future_shift=self.future_shift,
        ).set_index(self.date_col)

        self.predict_cols = [f"{target}_future{i}" for i in range(1, future_shift+1)]
        self.train_cols = [self.product_col] + [f"{target}_lag{i}" for i in range(1, lag_shift+1)]
        self.train_df[self.product_col] = self.train_df[self.product_col].astype("category")

        self.X = self.train_df[self.train_cols]
        self.y = self.train_df[self.predict_cols]

    def fit(self) -> None:
        """模型訓練."""
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(nested=True):
            self.model = XGBRegressor(**self.parameters)
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=12)
            self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

            # model parameters and model info
            model_params = self.model.get_xgb_params()
            mlflow.log_params(model_params)
            signature = infer_signature(x_train, self.model.predict(x_train))
            # save model
            mlflow.sklearn.log_model(
                self.model, LOG_MODEL_NAME, signature=signature,
            )
            # model validation info
            self.eval_result = self.model.evals_result()
            for metric_name, metric_values in self.eval_result["validation_0"].items():
                for step, value in enumerate(metric_values):
                    mlflow.log_metric(metric_name, value, step=step)

            errors = self.y - self.model.predict(self.X)
            self.errors = errors.rename(
                columns=lambda col: col.replace("future", "error"),
                ).assign(product_id=self.X[self.product_col])
            self.test_df = self.y.assign(product_id=self.X[self.product_col])
            self.pred_df = (
                pd.DataFrame(self.model.predict(self.X), index=self.y.index, columns=self.predict_cols)
                .assign(product_id=self.X[self.product_col])
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
        tmp = seasonal_product_df.reset_index(drop=True)[[self.date_col, self.target]].copy()
        time_series_tmp = tmp.set_index(self.date_col).squeeze()
        fill_value = time_series_tmp.describe()["25%"]
        time_series_tmp[time_series_tmp < 0] = fill_value
        time_series_tmp.index = pd.DatetimeIndex(time_series_tmp.index, freq="MS")
        stlf = STLForecast(time_series_tmp, ExponentialSmoothing, model_kwargs={"trend": True}, robust=True)
        es_model = stlf.fit()
        es_predict = es_model.forecast(forecast_steps)
        es_predict[es_predict < 0] = fill_value
        return es_predict.reset_index().rename(columns={"index": self.date_col, 0: self.target})

    def _quantile_result(self, forecast_df: pd.DataFrame, target: str) -> pd.DataFrame:
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
            train_tmp = self.train_df[self.train_df[self.product_col] == pid].copy()
            # 針對非季節性商品進行 multi-output 預測
            if pid not in seasonal_product_id:
                tmp_x = np.array(train_tmp[self.X.columns]).tolist()
                tmp_y = np.array(train_tmp[self.y.columns]).tolist()

                for _i in range(n_periods):

                    input_seq = tmp_x[-1].copy()
                    input_seq[self.future_shift+1:] = input_seq[len(self.product_col):(self.lag_shift-self.future_shift)+1]
                    input_seq[len(self.product_col):self.future_shift+1] = tmp_y[-1][::-1]
                    tmp_x.append(input_seq)

                    input_seq_df = pd.DataFrame([input_seq], columns=self.X.columns).astype({self.product_col: "category"})
                    self.input_seq_df = input_seq_df

                    tmp_yi = self.model.predict(self.input_seq_df)[0]
                    tmp_yi = np.where(tmp_yi > np.max(tmp_y), np.max(tmp_y), tmp_yi)
                    tmp_yi = np.where(tmp_yi < np.min(tmp_y), np.min(tmp_y), tmp_yi)
                    tmp_y.append(tmp_yi)

                last_date = train_tmp.index.max() + pd.DateOffset(months=self.future_shift-1)
                dates = pd.date_range(start=last_date, periods=n_periods*self.future_shift+1, freq="MS")
                self.dates = dates[dates != last_date]

                forecast_df = pd.DataFrame(
                    {self.date_col: self.dates},
                ).assign(
                    product_id=pid,
                    month=lambda df: df[self.date_col].dt.month,
                )

                # 利用權重修正預測資料
                tmp_y[-1][0] = tmp_y[-2][-1] * 0.2 + tmp_y[-1][0] * 0.8
                for i in range(1, len(tmp_y[-1])):
                    tmp_y[-1][i] = tmp_y[-1][i-1] * 0.2 + tmp_y[-1][i] * 0.8

                # 負數處理成正數
                tmp_y = np.maximum(tmp_y, 0)

                self.forecast_df = pd.concat(
                    (forecast_df, pd.concat(
                        [pd.Series(i, name=self.target) for i in tmp_y[-n_periods:]],
                    ).reset_index(drop=True),
                ), axis=1)

            # 如果是「季節性」商品,利用傳統時間序列滾動進行預測
            else:
                train_tmp = self.dataset[self.dataset[self.product_col] == pid].copy()
                ts_train_df = train_tmp.reset_index()[[self.date_col, self.target]]
                ts_train_df.columns = [self.date_col, self.target]
                forecast_df = self._seasonal_product_forecast(
                    seasonal_product_df=ts_train_df,
                    forecast_steps=n_periods*6, #n_periods*6=12筆
                )
                self.forecast_df = forecast_df.assign(
                    product_id=pid,
                    month=forecast_df[self.date_col].dt.month,
                )

            # quantile result
            self.forecast_df = self._quantile_result(self.forecast_df, target=self.target)

            forecast_dfs.append(self.forecast_df)
            logging.debug("forecast_dataframe: %s", self.forecast_df)

        return pd.concat(forecast_dfs, axis=0)
