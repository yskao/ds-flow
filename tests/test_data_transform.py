import pandas as pd
from mllib.data_transform import TransformDataForTraining
from pytest_mock import MockFixture

from utils.gcp.client import get_bigquery_client

bigquery_client = get_bigquery_client()


class TestTranformDataForTraining:
    """test transform."""

    def test_p02_get_transformed_training_data(self, mocker: MockFixture):

        expected_df = pd.DataFrame()
        test_transform = TransformDataForTraining(department_code="P02")

        mocker.patch.object(
            target=test_transform,
            attribute="get_transformed_training_data",
            return_value=expected_df,
        )

        output_df = test_transform.get_transformed_training_data(
            start_date="2023-01-01",
            end_date="2023-03-01",
            bigquery_client=bigquery_client,
        )

        assert all(output_df == expected_df)


    def test_p03_get_transformed_training_data(self, mocker: MockFixture):

        test_transform = TransformDataForTraining(department_code="P03")
        expected_df = pd.DataFrame()

        mocker.patch.object(
            target=test_transform,
            attribute="get_transformed_training_data",
            return_value=expected_df,
        )

        output_df = test_transform.get_transformed_training_data(
            start_date="2023-01-01",
            end_date="2023-03-01",
            bigquery_client=bigquery_client,
        )
        assert all(output_df == expected_df)


    def test_p04_get_transformed_training_data(self, mocker: MockFixture):
        test_transform = TransformDataForTraining(department_code="P04")
        expected_df = pd.DataFrame()

        mocker.patch.object(
            target=test_transform,
            attribute="get_transformed_training_data",
            return_value=expected_df,
        )

        output_df = test_transform.get_transformed_training_data(
            start_date="2023-01-01",
            end_date="2023-03-01",
            bigquery_client=bigquery_client,
        )
        assert all(output_df == expected_df)
