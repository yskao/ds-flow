import pandas as pd
from mllib.get_product_target import GetProductTargetForP02
from pytest_mock import MockFixture

get_product_target_for_p02 = GetProductTargetForP02()


class TestGetProductTargetForP02:
    """test get product_target_for_p02."""

    def test_get_training_target(self, mocker: MockFixture):
        expected_df = pd.DataFrame()
        mock_get_p02_training_target = mocker.patch("mllib.get_product_target.get_p02_training_target")
        mock_get_p02_training_target.return_value = expected_df
        get_product_target_for_p02 = GetProductTargetForP02()
        output_df = get_product_target_for_p02.get_training_target()
        assert all(output_df == expected_df)
        mock_get_p02_training_target.assert_called_once()


    def test_get_output_df(self, mocker: MockFixture):
        expected_df = pd.DataFrame()
        mocker.patch.object(
            get_product_target_for_p02,
            "get_output_df",
            return_value=pd.DataFrame(),
        )
        output_df = get_product_target_for_p02.get_output_df()
        assert all(output_df == expected_df)
