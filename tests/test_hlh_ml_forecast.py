
import numpy as np
import pandas as pd
from mllib.hlh_ml_forecast import HLHMLForecast


def test_hlh_ml_forecast(p03_sales_data_train_df, p03_predict_result):

    expected_single_predict = pd.DataFrame(
        [108.975136, 67.039856, 117.11654, 180.88374, 153.96512, 173.00777]).T
    expected_df = p03_predict_result

    model = HLHMLForecast(
        dataset=p03_sales_data_train_df,
        target="sales",
        n_estimators=200,
    )
    model.fit()

    single_predict = pd.DataFrame(model.predict(model.X[:1]))
    assert np.allclose(expected_single_predict, single_predict)

    product_id_list = p03_sales_data_train_df["product_id_combo"].unique()
    predict_df = model.rolling_forecast(n_periods=1, product_id=product_id_list).reset_index(drop=True)

    assert all(predict_df["date"] == expected_df["date"])
    assert all(predict_df["product_id_combo"] == expected_df["product_id_combo"])
    assert all(predict_df["month"] == expected_df["month"])
    assert (np.allclose(predict_df["sales"], expected_df["sales"]))
    assert (np.allclose(predict_df["less_likely_lb"], expected_df["less_likely_lb"]))
    assert (np.allclose(predict_df["likely_lb"],expected_df["likely_lb"]))
    assert (np.allclose(predict_df["likely_ub"],expected_df["likely_ub"]))
    assert (np.allclose(predict_df["less_likely_ub"],expected_df["less_likely_ub"]))
