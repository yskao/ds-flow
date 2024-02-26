"""sale forecast flow."""
from typing import TypeVar

import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from mllib.data_transform import TransformDataForTraining
from mllib.forecasting.sales_forecasting.hlh_ml_forecast import HLHMLForecast
from mllib.ml_utils.utils import (
    transform_data_for_reference,
)
from mllib.sql_query.sales_forecasting_p03_script import (
    create_p03_model_predict,
    create_p03_model_referenceable,
    create_p03_model_testing,
)
from prefect import flow, task

from utils.gcp.client import get_bigquery_client
from utils.logging_udf import get_logger
from utils.prefect import generate_flow_name
from utils.tools.common import get_agent_forecast_data, get_net_month_qty_p03_data

Predictor = TypeVar("Predictor")
dep_code = "P03"
trans_model = TransformDataForTraining(department_code=dep_code)
logger = get_logger(in_prefect=True)

@task(name="create_p03_model_testing_table")
def create_p03_model_testing_table(bigquery_client: BigQueryClient) -> None:
    """Create src_ds.p03_model_testing."""
    bigquery_client.query(create_p03_model_testing()).result()


@task(name="create_p03_model_predict_table")
def create_p03_model_predict_table(bigquery_client: BigQueryClient) -> None:
    """Create src_ds.p03_model_predict."""
    bigquery_client.query(create_p03_model_predict()).result()


@task(name="create_p03_model_referenceable_table")
def create_p03_model_referenceable_table(bigquery_client: BigQueryClient) -> None:
    """Create src_ds.p03_model_referenceable."""
    bigquery_client.query(create_p03_model_referenceable()).result()


@task(name="prepare_training_data")
def prepare_training_data(
    start_date: str,
    end_date: str,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """從指定的日期區間取得三處每月淨銷量數據."""
    logger.info("data extracting ...")
    training_data = get_net_month_qty_p03_data(
        start_date=start_date,
        end_date=end_date,
        bigquery_client=bigquery_client,
    )
    logger.info("data extracting finish")
    return training_data


@task(name="model_training")
def train_model(train_df: pd.DataFrame) -> Predictor:
    """使用指定的訓練數據訓練模型。."""
    logger.info("model training ...")
    model = HLHMLForecast(
        dataset=train_df,
        date_col="order_ym",
        product_col="product_custom_id",
        target="net_sale_qty",
        n_estimators=400,
    )
    model.fit()
    logger.info("model training finish")
    return model


@task(name="model_predicting")
def model_predict(model: Predictor, train_df: pd.DataFrame) -> pd.DataFrame:
    """使用指定的模型和訓練數據對未來進行預測。."""
    logger.info("model predicting ...")
    product_id_list = train_df["product_custom_id"].unique()
    predict_df = model.rolling_forecast(n_periods=2, product_id=product_id_list)
    logger.info("model predicting finish")
    return predict_df


@task(name="generate testing table")
def generate_model_testing_df(
    model: Predictor,
    target_time: str,
    train_df: pd.DataFrame,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """模型測試表."""
    logger.info("generating model testing table ...")
    target_time = pd.to_datetime(target_time)
    start_month = (target_time - pd.DateOffset(months=12)).strftime("%Y-%m-01")

    # 業務預估取當月預估值即可 gap=1
    agent_forecast_data = get_agent_forecast_data(
        start_date=start_month,
        end_date=target_time,
        bigquery_client=bigquery_client,
    ).query("estimate_month_gap==1")

    # test_table 取預測版本最新的一期
    test_data = model.test_df.custom_data_for_reference(_cols="sales").query("predicted_on_date == predicted_on_date.max()")

    pred_table = model._quantile_result(
        transform_data_for_reference(test_df=model.pred_df, _cols="sales")
        .rename(columns={"sales": "sales_model"}),
        target="sales_model",
    )

    # test_df = (
    #     test_table
    #     .merge(agent_forecast, on=["date", "product_id_combo"], how="left")
    #     .merge(train_df, on=["sales", "date", "product_id_combo"], how="left")
    #     .merge(pred_table, on=["predicted_on_date", "product_id_combo", "date", "M"])
    #     .merge(training_info[["product_name", "product_id_combo"]], on="product_id_combo", how="left")
    #     .assign(
    #         sales_agent=lambda df: df["sales_agent"].fillna(0),
    #         positive_ind_likely=lambda df: np.where((df["sales"] >= df["likely_lb"]) & (df["sales"] <= df["likely_ub"]), 1, 0),
    #         positive_ind_less_likely=lambda df: np.where((df["sales"] >= df["less_likely_lb"])
    #                                                      & (df["sales"] <= df["less_likely_ub"])
    #                                                      & (df["positive_ind_likely"] != 1), 1, 0),
    #     )
    #     .drop(columns=["year_weight"])
    #     .sort_values(["product_id_combo", "M"])
    # )
    return test_data


@task(name="generate reference table")
def generate_reference_table(
    model: Predictor,
    target_time: str,
    train_df: pd.DataFrame,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """生成「高參考-低參考」表。."""
    # 製作「高參考-低參考」表
    logger.info("generating reference table ...")
    target_time = pd.to_datetime(target_time)
    start_month = (pd.Timestamp.now("Asia/Taipei") - pd.DateOffset(months=6)).strftime("%Y-%m-01")
    agent_forecast = trans_model.data_extraction.get_agent_forecast_data(
        start_month=start_month,
        end_month=target_time,
        training_info = trans_model.data_extraction.get_training_target(),
        bigquery_client=bigquery_client,
    )
    test_table = custom_data_for_reference(test_df=model.errors)
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
    # 計算好的測試資料存入 psi.f_model_testing
    logger.info("store test to db ...")
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
    # 預測好的資料存入 psi.f_model_predict
    # P02的 training_info 欄位只有 product_id_combo、SPU,需要更改
    training_info = (
        trans_model.data_extraction.get_training_target()
        [["product_name", "product_id_combo"]]
    )
    product_info = train_df.merge(training_info, on="product_id_combo").drop(columns=["date"])
    logger.info("store predictions to db ...")
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
def mlops_sales_dep3_forecasting_flow(init: bool=False) -> None:

    bigquery_client = get_bigquery_client()
    end_date = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-01")

    if init:
        logger.info("create_p03_model_predict_table ...")
        create_p03_model_predict_table(bigquery_client)
        logger.info("create_p03_model_testing_table ...")
        create_p03_model_testing_table(bigquery_client)
        logger.info("create_p03_model_referenceable_table ...")
        create_p03_model_referenceable_table(bigquery_client)

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
        bq_table="src_ds.ds_p03_model_testing",
        department_code=dep_code,
        bigquery_client=bigquery_client,
    )
    store_predictions_to_bq(
        train_df=train_df,
        predict_df=predict_df,
        bq_table="src_ds.ds_p03_model_predict",
        department_code=dep_code,
        bigquery_client=bigquery_client,
    )
    store_references_to_bq(
        mae_df=reference_df,
        bq_table="src_ds.ds_p03_model_referenceable",
        department_code=dep_code,
        bigquery_client=bigquery_client,
    )


if __name__ == "__main__":
    mlops_sales_dep3_forecasting_flow(False)
