import pandas as pd
from mllib.data_engineering import prepare_predict_table_to_sql


def test_prepare_predict_table_to_sql(p03_training_info, p03_predict_result):
    pd.DataFrame()
    prepare_predict_table_to_sql(
        predict_df=p03_predict_result,
        product_data_info=p03_training_info,
        predicted_on_date="2023-06-01",
        department_code="P03",
    )
