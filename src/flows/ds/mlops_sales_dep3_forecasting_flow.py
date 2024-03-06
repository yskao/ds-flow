"""sale forecast flow."""
import shutil
from pathlib import Path
from typing import TypeVar

import mlflow
import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.storage import Client as GCSClient
from mllib.forecasting.sales_forecasting.hlh_ml_forecast import HLHMLForecast
from prefect import flow, task

from utils.forecasting.sales_forecasting.sql import (
    create_p03_model_predict,
    create_p03_model_referenceable,
    create_p03_model_testing,
)
from utils.forecasting.sales_forecasting.utils import (
    gen_model_testing_df,
    load_seasonal_product_ids,
    predict_and_test_data_to_bq,
    reference_data_to_bq,
    update_artifact_location,
)
from utils.gcp.client import get_bigquery_client, get_gcs_client
from utils.logging_udf import get_logger
from utils.prefect import generate_flow_name
from utils.tools.common import (
    get_net_month_qty_p03_data,
    upload_directory_to_gcs,
)

DEPT_CODE = "P03"
Predictor = TypeVar("Predictor")
logger = get_logger(in_prefect=True)
CURRENT_PATH = Path(__file__).parent.parent.parent
BUCKET_NAME = "ml-project-hlh"
EXPERIMENT_NAME = "Sales-Forecasting-P03"
seasonal_product_list = load_seasonal_product_ids(
    f"{CURRENT_PATH}/utils/forecasting/sales_forecasting/seasonal_product.yml")


@task
def create_p03_model_testing_table(bigquery_client: BigQueryClient) -> None:
    """Create src_ds.p03_model_testing."""
    bigquery_client.query(create_p03_model_testing()).result()


@task
def create_p03_model_predict_table(bigquery_client: BigQueryClient) -> None:
    """Create src_ds.p03_model_predict."""
    bigquery_client.query(create_p03_model_predict()).result()


@task
def create_p03_model_referenceable_table(bigquery_client: BigQueryClient) -> None:
    """Create src_ds.p03_model_referenceable."""
    bigquery_client.query(create_p03_model_referenceable()).result()


@task
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


@task
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


@task
def model_predict(model: Predictor, train_df: pd.DataFrame) -> pd.DataFrame:
    """使用指定的模型和訓練數據對未來進行預測。."""
    logger.info("model predicting ...")
    product_id_list = train_df["product_custom_id"].unique()
    predict_df = model.rolling_forecast(
        n_periods=2,
        product_id=product_id_list,
        seasonal_product_id=seasonal_product_list,
    )
    logger.info("model predicting finish")
    return predict_df


@task
def generate_testing_table(
    model: Predictor,
    target_time: str,
    bigquery_client: BigQueryClient,
) -> pd.DataFrame:
    """模型測試表."""
    logger.info("generating model testing table ...")
    final_df, predict_df, reference_df = gen_model_testing_df(
        model=model,
        target_time=target_time,
        bigquery_client=bigquery_client,
    )
    logger.info("generating model testing table finish")
    return final_df, predict_df, reference_df


@task
def store_test_to_bq(
    test_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> None:
    """將測試結果存儲到資料庫中。."""
    # 計算好的測試資料存入 psi.f_model_testing
    logger.info("store test to bq ...")
    predict_and_test_data_to_bq(
        df_for_upload=test_df,
        bq_table=bq_table,
        department_code=department_code,
        bigquery_client=bigquery_client,
    )
    logger.info("store test to bq finish")


@task
def store_metadata_and_artifacts_to_gcs(
    bucket_name: str,
    source_dir: str,
    destination_dir: str,
    gcs_client: GCSClient,
) -> None:
    """Upload file to gcs."""
    logger.info("store metadata and model to gcs ...")
    update_artifact_location(
        root_dir=source_dir,
        update_artifact_location=destination_dir,
    )
    upload_directory_to_gcs(
        bucket_name=bucket_name,
        source_dir=source_dir,
        destination_dir=destination_dir,
        gcs_client=gcs_client,
    )
    remove_path = Path(source_dir)
    if remove_path.exists() and remove_path.is_dir():
        shutil.rmtree(remove_path)
    logger.info("store metadata and model to gcs finish")


@task
def store_predictions_to_bq(
    predict_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> None:
    """將預測結果存儲到數據庫中。."""
    # 預測好的資料存入 psi.f_model_predict
    logger.info("store predictions to bq ...")
    predict_and_test_data_to_bq(
        df_for_upload=predict_df,
        bq_table=bq_table,
        department_code=department_code,
        bigquery_client=bigquery_client,
    )
    logger.info("store predictions to bq finish")


@task
def store_references_to_bq(
    reference_df: pd.DataFrame,
    bq_table: str,
    department_code: str,
    bigquery_client: BigQueryClient,
) -> None:
    """將「高參考-低參考」表存儲到數據庫中。."""
    # 計算好的高參考低參考資料存入 psi.f_model_referenceable
    logger.info("store references to bq ...")
    reference_data_to_bq(
        reference_df=reference_df,
        bq_table=bq_table,
        department_code=department_code,
        bigquery_client=bigquery_client,
    )
    logger.info("store references to bq finish")


@flow(name=generate_flow_name())
def mlops_sales_dep3_forecasting_flow(init: bool=False) -> None:
    """Main flow."""
    bigquery_client = get_bigquery_client()
    gcs_client = get_gcs_client()
    end_date = pd.Timestamp.now("Asia/Taipei").strftime("%Y-%m-01")
    metadata_version_date = end_date.replace("-", "")
    metadata_path = "mlruns/" + f"{metadata_version_date}/" + "mlruns/"

    # setting mlflow tracking path which must be same as download path from gcs
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.set_tracking_uri(uri=metadata_path)

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
    test_df, predict_df, reference_df = generate_testing_table(
        model=forecast_model,
        target_time=end_date,
        bigquery_client=bigquery_client,
    )
    store_metadata_and_artifacts_to_gcs(
        bucket_name=BUCKET_NAME,
        source_dir=metadata_path,
        destination_dir=metadata_path,
        gcs_client=gcs_client,
    )
    store_test_to_bq(
        test_df=test_df,
        bq_table="src_ds.ds_p03_model_testing",
        department_code=DEPT_CODE,
        bigquery_client=bigquery_client,
    )
    store_predictions_to_bq(
        predict_df=predict_df,
        bq_table="src_ds.ds_p03_model_predict",
        department_code=DEPT_CODE,
        bigquery_client=bigquery_client,
    )
    store_references_to_bq(
        reference_df=reference_df,
        bq_table="src_ds.ds_p03_model_referenceable",
        department_code=DEPT_CODE,
        bigquery_client=bigquery_client,
    )


if __name__ == "__main__":
    mlops_sales_dep3_forecasting_flow(False)
