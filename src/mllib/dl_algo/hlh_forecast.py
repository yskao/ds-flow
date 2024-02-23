import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

torch.manual_seed(0)

class LSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        layer_size: int,
    ) -> None:
        """
        初始化 LSTMNet 模型.

        Args:
        ----
        - input_size (int): 模型輸入特徵數
        - hidden_size (int): 隱藏層維度大小
        - output_size (int): 模型輸出大小
        - layer_size (int): LSTM 層數

        Returns:
        -------
        - None

        """
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_size = layer_size

        # batch_first --> batch_dim, seq_dim, feature_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_size,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        模型預測.

        Args:
        ----
        - sequence (torch.Tensor): 輸入序列

        Returns:
        -------
        - y_pred (torch.Tensor): 模型輸出序列

        """
        # layer_size, 1 --> 指的是每次輸入的資料 batch_size=1 (1 表示一包資料長度為 time_window)
        h0 = torch.zeros(self.layer_size, 1, self.hidden_size)
        c0 = torch.zeros(self.layer_size, 1, self.hidden_size)
        lstm_out, (_, _) = self.lstm(
            sequence,
            (h0.detach(), c0.detach()),
        )
        y_pred = self.fc(lstm_out[:, -1, :])
        return y_pred


class HLHForecast:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        hidden_size: int,
        time_window: int,
        epochs: int,
        interval_time: int,
        sample_weight_feature: str | None=None,
        layer_size: int=1,
        forecast_size: int=1,
    ) -> None:
        """
        初始化函式,設定模型的參數、資料、權重等相關參數。.

        參數:
        ---------
        dataframe : pd.DataFrame
            包含需預測目標的 pandas DataFrame。
        target : str
            預測目標的 column 名稱。
        hidden_size : int
            LSTM 模型中的 hidden size。
        time_window : int
            前幾天的歷史資料用來訓練模型。
        epochs : int
            訓練模型的迭代次數。
        interval_time : int
            預測的時間間隔(單位為月)。
        sample_weight_feature : Optional[str], default=None
            可選權重特徵,用來平衡資料集。
        layer_size : int, default=1
            LSTM 模型中的 layer size。
        forecast_size : int, default=1
            預測的期數。

        範例:
        -----
        >>> hlh_forecast = HLHForecast(
        ...     dataframe=df,
        ...     target='sales',
        ...     hidden_size=20,
        ...     time_window=14,
        ...     epochs=100,
        ...     interval_time=1,
        ... )

        """
        self.data = dataframe
        self.target = target
        self.sample_weight = (
              pd.Series(np.ones(len(dataframe)))
              if sample_weight_feature is None
              else dataframe[sample_weight_feature]
          )
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.time_window = time_window
        self.num_epochs = epochs
        self.interval_time = interval_time
        self.output_size = forecast_size

        error_msg = "epochs must be 1 when time_window = length of data"
        if len(dataframe) == time_window and epochs != 1:
            raise ValueError(error_msg)

        logging.debug(f"hidden_size:{hidden_size}, time_window:{time_window}" f" num_epochs:{epochs}") # noqa


    @staticmethod
    def split_data(
        sequence: np.ndarray, time_window: int, output_y: int=1,
      ) -> list[tuple[Any, Any]]:
        """
        拆分資料為輸入序列和輸出序列,用於 LSTM 模型的訓練。.

        參數:
        ---------
        sequence : np.ndarray
            需要拆分的序列。
        time_window : int
            前幾天的歷史資料用來訓練模型。
        output_y : int, default=1
            輸出序列的長度。

        返回:
        -------
        Tuple[Any, Any]
            輸入和輸出序列所組成的 tuple。

        範例:
        -----
        >>> input_seq = np.array([1, 2, 3, 4, 5])
        >>> time_window = 3
        >>> output_y = 2
        >>> HLHForecast.split_data(input_seq, time_window, output_y)
        [(array([1, 2, 3]), array([4, 5])),
        (array([2, 3, 4]), array([5, 6])),
        (array([3, 4, 5]), array([6, 7]))]
        """
        tuple_genereated_sequence = []
        for i in range(len(sequence) - time_window):
            x = sequence[i : i + time_window]
            y = sequence[i + time_window : i + time_window + output_y]
            tuple_genereated_sequence.append((x, y))

        return tuple_genereated_sequence


    def __prepare_data(self) -> list[tuple[Any, Any]]:
        """
        將原始資料進行預處理,包括將目標欄位轉為 float,進行歸一化,並進行訓練數據的切割。.

        Returns
        -------
            List[Tuple[Any, Any]]: 切割後的訓練數據,為一個 Tuple 組成的列表,每個 Tuple 包括一個時間窗口的訓練數據和對應的標籤。

        """
        self.train_data = self.data[[self.target]].values.astype(float)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_scaled = self.scaler.fit_transform(self.train_data.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_scaled).view(-1)  # converting to Tensor
        self.train_data_normalized = train_data_normalized
        return self.split_data(
            sequence=self.train_data_normalized,
            time_window=self.time_window,
            output_y=self.output_y,
        )


    def fit(self, **kwargs: float) -> None:
        """
        使用給定的超參數進行訓練。訓練過程中會進行訓練數據的預處理、模型初始化、損失函數與優化器的設置以及模型訓練。最後會保存模型以及訓練時的一些結果供後續使用。.

        Args:
        ----
            **kwargs: 包含超參數的字典,可以包含以下參數:
                lr (float): 優化器的學習率,預設為 0.001。
                output_y (int): 預測的時間長度,預設為 1。

        Returns:
        -------
            None

        """
        logging.debug(f"kwargs:{kwargs}") # noqa
        self.lr = kwargs.get("lr", 0.001)
        self.output_y = kwargs.get("output_y", 1)

        model = LSTMNet(
            input_size=1, # feature_dim
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            layer_size=self.layer_size,
        )
        self.model = model

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.pred_training_list = []
        self.hist = np.zeros(self.num_epochs) # for plot
        self.train_input_output = self.__prepare_data()
        for i in range(int(self.num_epochs)):
            for idx, (input_seq, target) in enumerate(self.train_input_output):
                logging.debug("input_seq %s", input_seq.view(1, input_seq.size(0), 1).size())
                optimizer.zero_grad()
                # number_layer, batch, hidden
                y_pred = model(input_seq.view(1, input_seq.size(0), 1))[0]
                loss = criterion(y_pred, target) * self.sample_weight.iloc[idx]
                if i == int(self.num_epochs)-1:
                    self.pred_training_list.append(y_pred.detach().numpy())
                loss.backward()
                optimizer.step()
            self.hist[i] = loss
            if i % 25 == 1:
                logging.info(f'epoch: {i:3}, loss: {loss.item():10.8f}') # noqa
        self.pred_in_training = self.scaler.inverse_transform(
            np.array(self.pred_training_list).reshape(-1,1),
        ).flatten()


    def rolling_forecast(self, n_periods: int) -> pd.DataFrame:
        """
        根據已訓練好的模型,進行指定時間長度的預測。預測過程中會使用已保存的模型,並根據模型訓練時的預處理方法對預測數據進行還原,得到實際預測值。.

        Args:
        ----
            n_periods (int): 需要預測的時間長度。

        Returns:
        -------
            pd.DataFrame: 預測值的 DataFrame,其中包含時間序列與預測值。預測值已還原為原始數據的尺度,並已按照時間序列排序。

        """
        model = self.model
        scaler = self.scaler

        error_msg = "fit() function is a must"
        if model is None:
            raise ValueError(error_msg)

        train_data_normalized = self.train_data_normalized
        assert train_data_normalized is not None and scaler is not None # noqa
        time_window = self.time_window

        model.eval()  # Sets the module in evaluation mode

        test_inputs = train_data_normalized[-time_window:].tolist()  # last train sequence
        for _ in range(n_periods):
            input_seq = torch.FloatTensor(test_inputs[-time_window:])
            with torch.no_grad():
                test_inputs.append(model(input_seq.view(-1, input_seq.size(0), 1)).item())
        inverse_normalized_data = scaler.inverse_transform(
            np.array(test_inputs[time_window:]).reshape(-1, 1),
          ).flatten()

        logging.info("model generated forecast values: %s", inverse_normalized_data)

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=str(self.interval_time)+"MS")
        self.dates = dates[dates != last_date]
        self.forecast_df = pd.DataFrame(
            {"date": self.dates, self.target: inverse_normalized_data},
          ).assign(month=lambda df: df["date"].dt.month)
        logging.info("forecast_dataframe: %s", self.forecast_df)

        return self.forecast_df.set_index("date").round(3)
