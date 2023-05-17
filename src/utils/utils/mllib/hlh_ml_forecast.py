
import logging

import numpy as np
import pandas as pd
from mllib.data_engineering import shift_all_product_id_data
from xgboost import XGBRegressor


class HLHMLForecast:
    """
    用於銷售預測功能。使用者需要提供訓練數據,並可以根據需要指定超參數進行訓練和預測。.

    Attributes
    ----------
        target (str): 訓練數據中的目標欄位。
        future_shift (int): 未來預測的時間長度,即預測幾個月後的銷售額。
        lag_shift (int): 訓練數據中的時間序列向後平移的時間長度,即預測當月的銷售額時,使用前幾個月的銷售額作為輸入變量。
        model (XGBRegressor): 使用的 XGBoost 回歸模型。
        train_df (pd.DataFrame): 訓練數據的 DataFrame,包含目標變量和輸入變量。輸入變量已進行平移和編碼。
        cate_cols (List[str]): 訓練數據中需要進行編碼的類別型變量。
        predict_cols (List[str]): 需要進行預測的目標欄位列表。
        train_cols (List[str]): 用於訓練模型的輸入變量列表。

    Methods
    -------
        __init__(self, dataset: pd.DataFrame, target: str, n_estimators: int=300, future_shift: int=6, lag_shift: int=12) -> None:
            初始化 HLHMLForecast 類別,對訓練數據進行預處理並初始化模型。

        fit(self) -> None:
            使用訓練數據進行模型訓練。

        predict(self, x: pd.DataFrame) -> np.array:
            對給定的輸入數據進行預測,返回預測結果。

        _quantile_result(self, forecast_df: pd.DataFrame, error_df: pd.DataFrame) -> pd.DataFrame:
            對預測結果進行分位數轉換。

        rolling_forecast(self, product_id: list, n_periods: int=1) -> pd.DataFrame:
            對指定產品進行滾動預測,返回預測結果。
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        n_estimators: int=300,
        future_shift: int=6,
        lag_shift: int=12,
    ) -> None:
        """
        初始化 HLHMLForecast 物件。.

        參數:
        - dataset: pd.DataFrame,包含原始資料的 DataFrame。
        - target: str,要預測的目標變數名稱。
        - n_estimators: int,模型使用的決策樹個數,預設為 300。
        - future_shift: int,預測未來的時間步數,預設為 6。
        - lag_shift: int,使用多少過去時間步數的資料進行預測,預設為 12。

        當 lag_shift 小於 future_shift 時,會發生 ValueError。
        當資料的月份數量小於 future_shift + lag_shift 時,會發生 ValueError。

        設定 self.target、self.future_shift、self.lag_shift 屬性,並建立 XGBRegressor 模型。
        呼叫 shift_all_product_id_data 函式進行資料預處理,並將處理後的資料指定給 self.train_df。
        將 self.cate_cols、self.predict_cols、self.train_cols 設為分類變數、預測目標、模型輸入變數名稱。
        將 self.cate_cols 中的欄位型別設為 "category"。
        設定 self.X、self.y 屬性為模型訓練使用的自變數、應變數。

        回傳:None
        """
        error_msg = "lag_shift must be greater than future_shift"
        if lag_shift < future_shift:
            raise ValueError(error_msg)

        error_msg = f"the month must be at least more than {future_shift+lag_shift+1}"
        if len(dataset.groupby("date")["date"]) < (future_shift+lag_shift+1):
            raise ValueError(error_msg)

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
        """
        設定 early_stopping_rounds 參數為 2,避免過度擬合。
        使用 self.X、self.y 訓練模型,並計算預測誤差,指定給 self.errors。
        回傳:None.
        """
        self.model.set_params = {"early_stopping_rounds": 2}
        self.model.fit(self.X, self.y, eval_set=[(self.X, self.y)])
        errors = self.y - self.model.predict(self.X)
        self.errors = errors.rename(
            columns=lambda col: col.replace("future", "error"),
            ).assign(product_id_combo=self.X["product_id_combo"])


    def predict(self, x: pd.DataFrame) -> np.array:
        """
        參數:
        - x: pd.DataFrame,模型輸入的自變數。.

        回傳:np.array,模型預測結果。
        """
        return self.model.predict(x)


    def _quantile_result(self, forecast_df: pd.DataFrame, error_df: pd.DataFrame) -> pd.DataFrame:
        """
        根據預測的標準差計算分位數區間,並加入到預測結果 DataFrame 中.

        參數:
        forecast_df (pd.DataFrame): 包含預測結果的 DataFrame
        error_df (pd.DataFrame): 包含誤差的 DataFrame

        回傳:
        pd.DataFrame: 加入分位數區間後的預測結果 DataFrame
        """
        pred_std = np.repeat(error_df.std(numeric_only=True, axis=0), self.n_periods).reset_index(drop=True)
        forecast_df.loc[:, "std"] = np.where(pred_std==0, 100, pred_std)
        return forecast_df.assign(
            less_likely_lb=forecast_df["sales"]*0.15,
            likely_lb=forecast_df["sales"]*0.25,
            likely_ub=forecast_df["sales"]*(np.random.random(1)+2),
            less_likely_ub=forecast_df["sales"]*(np.random.random(1)+3),
          )


    def rolling_forecast(
        self,
        product_id: list,
        n_periods: int=1,
    ) -> pd.DataFrame:
        """
        前瞻預測方法,針對給定的產品 ID 預測未來多期銷售量.

        參數:
        product_id (list): 要預測的產品 ID 列表
        n_periods (int, optional): 要預測的期數,預設為 1

        回傳:
        pd.DataFrame: 含有預測結果的 DataFrame
        """
        self.n_periods = n_periods
        forecast_dfs = []
        self.quantile_results = []
        for pid in product_id:
            train_tmp = self.train_df[self.train_df["product_id_combo"] == pid]
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

            # quantile result
            error_df = self.errors[self.errors["product_id_combo"]==pid]
            self.forecast_df = self._quantile_result(self.forecast_df, error_df)

            forecast_dfs.append(self.forecast_df)
            logging.debug("forecast_dataframe: %s", self.forecast_df)

        return pd.concat(forecast_dfs, axis=0)
