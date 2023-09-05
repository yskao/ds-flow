from typing import TypeVar

import numpy as np
import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from mllib.data_transform import TransformDataForTraining
from mllib.hlh_ml_forecast import HLHMLForecast
from mllib.ml_utils.sales_forecasting_dep4_utils import (
    dep04_seasonal_product_list,
    predict_data_to_bq,
    reference_data_to_bq,
    test_data_to_bq,
)
from mllib.ml_utils.utils import (
    get_mae_diff,
    get_test_data_for_reference,
)
from mllib.sql_query.sales_forecasting_p04_script import (
    create_p04_model_predict,
    create_p04_model_referenceable,
    create_p04_model_testing,
)
from prefect import flow, get_run_logger, task

from utils.gcp.client import get_bigquery_client
from utils.prefect import generate_flow_name

Predictor = TypeVar("Predictor")
dep_code = "P04"
trans_model = TransformDataForTraining(department_code=dep_code)


@task(name="create_p04_model_testing_table")
def create_p04_model_testing_table(bigquery_client: BigQueryClient) -> None:
    """Create DS.p04_model_testing."""
    bigquery_client.query(create_p04_model_testing()).result()


@task(name="create_p04_model_predict_table")
def create_p04_model_predict_table(bigquery_client: BigQueryClient) -> None:
    """Create DS.p04_model_predict."""
    bigquery_client.query(create_p04_model_predict()).result()


@task(name="create_p04_model_referenceable_table")
def create_p04_model_referenceable_table(bigquery_client: BigQueryClient) -> None:
    """Create DS.p04_model_referenceable."""
    bigquery_client.query(create_p04_model_referenceable()).result()


@task(name="prepare training data")
def prepare_training_data(
    start_date: str,
    end_date: str,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """從指定的日期區間和部門代碼獲取轉換後的訓練數據。."""
    logging = logging = get_run_logger()
    logging.info("data extracting ...")
    training_data = trans_model.get_transformed_training_data(
        start_date=start_date,
        end_date=end_date,
        bigquery_client=bigquery_client,
    )
    logging.info("data extracting finish")
    return training_data


@task(name="train model")
def train_model(train_df: pd.DataFrame) -> Predictor:
    """使用指定的訓練數據訓練模型。."""
    logging = logging = get_run_logger()
    logging.info("model training ...")
    model = HLHMLForecast(dataset=train_df, target="sales", n_estimators=200)
    model.fit()
    logging.info("model training finish")
    return model


@task(name="prediction by model")
def model_predict(model: Predictor, train_df: pd.DataFrame) -> pd.DataFrame:
    """使用指定的模型和訓練數據對未來進行預測。."""
    logging = logging = get_run_logger()
    logging.info("model predicting ...")
    product_id_list = train_df["product_id_combo"].unique()
    predict_df = model.rolling_forecast(
        n_periods=2,
        product_id=product_id_list,
        seasonal_product_id=dep04_seasonal_product_list())
    logging.info("model predicting finish")
    return predict_df


@task(name="generate testing table")
def generate_model_testing_df(
    model: Predictor,
    target_time: str,
    train_df: pd.DataFrame,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """模型測試表."""
    logging = logging = get_run_logger()
    logging.info("generating model testing table ...")
    target_time = pd.to_datetime(target_time)
    start_month = (target_time - pd.DateOffset(months=12)).strftime("%Y-%m-01")
    training_info = trans_model.data_extraction.get_training_target()

    agent_forecast = trans_model.data_extraction.get_agent_forecast_data(
        start_month=start_month,
        end_month=target_time,
        training_info = training_info,
        bigquery_client=bigquery_client,
    ).query("M==1")[["date", "product_id_combo", "sales_agent"]] # 業務預估取當月預估值即可

    test_table = (
        get_test_data_for_reference(test_df=model.test_df, _cols="sales")
        .query("predicted_on_date == predicted_on_date.max()")
    ) # test_table 取預測版本最新的一期

    pred_table = model._quantile_result(
        get_test_data_for_reference(test_df=model.pred_df, _cols="sales")
        .rename(columns={"sales": "sales_model"}),
        target="sales_model",
    )

    test_df = (
        test_table
        .merge(agent_forecast, on=["date", "product_id_combo"], how="left")
        .merge(train_df, on=["sales", "date", "product_id_combo"], how="left")
        .merge(pred_table, on=["predicted_on_date", "product_id_combo", "date", "M"])
        .merge(training_info[["product_name", "product_id_combo"]], on="product_id_combo", how="left")
        .assign(
            sales_agent=lambda df: df["sales_agent"].fillna(0),
            positive_ind_likely=lambda df: np.where((df["sales"] >= df["likely_lb"]) & (df["sales"] <= df["likely_ub"]), 1, 0),
            positive_ind_less_likely=lambda df: np.where((df["sales"] >= df["less_likely_lb"])
                                                         & (df["sales"] <= df["less_likely_ub"])
                                                         & (df["positive_ind_likely"] != 1), 1, 0),
        )
        .drop(columns=["year_weight"])
        .sort_values(["product_id_combo", "M"])
    )
    return test_df


@task(name="generate reference table")
def generate_reference_table(
    model: Predictor,
    target_time: str,
    train_df: pd.DataFrame,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """生成「高參考-低參考」表。."""
    # 製作「高參考-低參考」表
    logging = logging = get_run_logger()
    logging.info("generating reference table ...")
    target_time = pd.to_datetime(target_time)
    start_month = (pd.Timestamp.now("Asia/Taipei") - pd.DateOffset(months=6)).strftime("%Y-%m-01")
    agent_forecast = trans_model.data_extraction.get_agent_forecast_data(
        start_month=start_month,
        end_month=target_time,
        training_info = trans_model.data_extraction.get_training_target(),
        bigquery_client=bigquery_client,
    )
    test_table = get_test_data_for_reference(test_df=model.errors)
    combined_df = (
        test_table
        .merge(agent_forecast, on=["M", "date", "product_id_combo"], how="left")
        .merge(train_df, on=["date", "product_id_combo"], how="left")
    )
    combined_df["sales_agent"] = combined_df["sales_agent"].fillna(0)
    mae_df = get_mae_diff(combined_df)[["product_id_combo","bound_01_flag"]]
    brand_df = train_df[["product_id_combo", "brand"]].drop_duplicates()
    return mae_df.merge(brand_df, how="left", on="product_id_combo")


@task(name="store test to bq")
def store_test_to_bq(
    test_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> None:
    """將測試結果存儲到資料庫中。."""
    logging = logging = get_run_logger()
    # 計算好的測試資料存入 psi.f_model_testing
    logging.info("store test to db ...")
    test_data_to_bq(
        test_df=test_df,
        bq_table=bq_table,
        department_code=department_code,
        bigquery_client=bigquery_client,
    )


@task(name="store prediction to bq")
def store_predictions_to_bq(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> None:
    """將預測結果存儲到數據庫中。."""
    logging = logging = get_run_logger()
    # 預測好的資料存入 psi.f_model_predict
    # P02的 training_info 欄位只有 product_id_combo、SPU,需要更改
    training_info = (
        trans_model.data_extraction.get_training_target()
        [["product_name", "product_id_combo"]]
    )
    product_info = train_df.merge(training_info, on="product_id_combo").drop(columns=["date"])
    logging.info("store predictions to db ...")
    predict_data_to_bq(
        predict_df=predict_df,
        product_info=product_info,
        bq_table=bq_table,
        department_code=department_code,
        bigquery_client=bigquery_client,
    )


@task(name="store reference to bq")
def store_references_to_bq(
    mae_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> None:
    """將「高參考-低參考」表存儲到數據庫中。."""
    # 計算好的高參考低參考資料存入 psi.f_model_referenceable
    reference_data_to_bq(
        mae_df=mae_df,
        bq_table=bq_table,
        department_code=department_code,
        bigquery_client=bigquery_client,
    )


@flow(name=generate_flow_name())
def mlops_sales_dep4_forecasting_flow(init: bool=False) -> None:

    logging = logging = get_run_logger()
    bigquery_client = get_bigquery_client()
    end_date = pd.Timestamp.now("Asia/Taipei").tz_localize(None).strftime("%Y-%m-01")

    if init:
        logging.info("create_p04_model_predict_table ...")
        create_p04_model_predict_table(bigquery_client)
        logging.info("create_p04_model_testing_table ...")
        create_p04_model_testing_table(bigquery_client)
        logging.info("create_p04_model_referenceable_table ...")
        create_p04_model_referenceable_table(bigquery_client)

    # start_date 從 2020-01-01 開始
    train_df = prepare_training_data(
        start_date="2020-01-01",
        end_date=end_date,
        bigquery_client=bigquery_client,
    )
    forecast_model = train_model(
        train_df=train_df,
    )
    predict_df = model_predict(
        model=forecast_model,
        train_df=train_df,
    )
    test_df = generate_model_testing_df(
        model=forecast_model,
        target_time=end_date,
        train_df=train_df,
        bigquery_client=bigquery_client,
    )
    reference_df = generate_reference_table(
        model=forecast_model,
        target_time=end_date,
        train_df=train_df,
        bigquery_client=bigquery_client,
    )
    store_test_to_bq(
        test_df=test_df,
        bq_table="DS.ds_p04_model_testing",
        department_code=dep_code,
        bigquery_client=bigquery_client,
    )
    store_predictions_to_bq(
        train_df=train_df,
        predict_df=predict_df,
        bq_table="DS.ds_p04_model_predict",
        department_code=dep_code,
        bigquery_client=bigquery_client,
    )
    store_references_to_bq(
        mae_df=reference_df,
        bq_table="DS.ds_p04_model_referenceable",
        department_code=dep_code,
        bigquery_client=bigquery_client,
    )


if __name__ == "__main__":
    mlops_sales_dep4_forecasting_flow(False)
