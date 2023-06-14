from mllib.data_extraction import ExtractDataForTraining
from pytest_mock import MockFixture

test_eaxtract = ExtractDataForTraining()


class TestExtractDataForTraining:
    def test_get_training_target(self, mocker: MockFixture):
        expected_columns = ["brand", "product_id_combo", "自訂品名", "業務自訂分類", "product_id"]
        output_columns = mocker.patch.object(
            target=test_eaxtract,
            attribute="get_training_target",
            new=["brand", "product_id_combo", "自訂品名", "業務自訂分類", "product_id"],
        )
        assert output_columns == expected_columns


    def test_get_product_categories(self, mocker: MockFixture):
        expected_columns = ["product_id", "cat_code_1", "cat_code_2", "cat_code_3"]
        output_columns = mocker.patch.object(
            target=test_eaxtract,
            attribute="get_product_categories",
            new=["product_id", "cat_code_1", "cat_code_2", "cat_code_3"],
        )
        assert (output_columns == expected_columns)


    def test_get_sales_data(self, mocker: MockFixture, p03_sales_data):
        use_cols = ["日期", "銷貨數量", "銷貨淨數量", "銷貨贈品數量", "銷貨金額", "銷貨淨金額", "品號", "事業處代號", "date",
       "month_begin_date"]
        expected_df = p03_sales_data[use_cols]
        output_df = mocker.patch.object(
            target=test_eaxtract,
            attribute="get_sales_data",
            new=p03_sales_data[use_cols],
        )
        assert all(output_df == expected_df)
