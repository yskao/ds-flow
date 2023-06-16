from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def p03_training_info():
    data_path = Path().resolve().joinpath("tests/test_data/p03_training_info.csv")
    return pd.read_csv(data_path)


@pytest.fixture()
def p04_training_info():
    data_path = Path().resolve().joinpath("tests/test_data/p04_training_info.csv")
    return pd.read_csv(data_path)


@pytest.fixture()
def p03_sales_predict_data():
    data_path = Path().resolve().joinpath("tests/test_data/ds_p03_model_predict_data.csv")
    return pd.read_csv(data_path, parse_dates=["predicted_on_date", "date"])


@pytest.fixture()
def p04_sales_predict_data():
    data_path = Path().resolve().joinpath("tests/test_data/ds_p04_model_predict_data.csv")
    return pd.read_csv(data_path, parse_dates=["predicted_on_date", "date"])


@pytest.fixture()
def p03_transformed_sales_data():
    data_path = Path().resolve().joinpath(
        "tests/test_data/p03_sales_forecasting_transformed_data_202301_202302.csv")
    return pd.read_csv(data_path, parse_dates=["date"])


@pytest.fixture()
def p04_transformed_sales_data():
    data_path = Path().resolve().joinpath(
        "tests/test_data/p04_sales_forecasting_transformed_data_202301_202302.csv")
    return pd.read_csv(data_path, parse_dates=["date"])


@pytest.fixture()
def p03_sales_data_train_df():
    data_path = Path().resolve().joinpath(
        "tests/test_data/p03_sales_forecasting_train_df_20210901_2023-0601.csv")
    return pd.read_csv(data_path, parse_dates=["date"])


@pytest.fixture()
def p03_predict_result():
    data_path = Path().resolve().joinpath(
        "tests/test_data/p03_predict_result.csv")
    return pd.read_csv(data_path, parse_dates=["date"])


@pytest.fixture()
def soda_stream_repurchase_data():
    data_path = Path().resolve().joinpath(
        "tests/test_data/test_soda_stream_repurchase/soda_stream_df.csv")
    data = pd.read_csv(
        data_path, parse_dates=["order_date"], dtype={"product_category_id_2": "str", "mobile": "category"})
    return data
