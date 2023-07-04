import pandas as pd
from mllib.data_engineering import (
    get_continuing_buying_for_the_past_two_years,
    prepare_predict_table_to_sql,
)


def test_prepare_predict_table_to_sql(p03_training_info, p03_predict_result):
    pd.DataFrame()
    prepare_predict_table_to_sql(
        predict_df=p03_predict_result,
        product_data_info=p03_training_info,
        predicted_on_date="2023-06-01",
        department_code="P03",
    )


def test_get_continuing_buying_for_the_past_two_years():
    data = {
        "mobile": [
            "0922111111",
            "0922111111",
            "0922111111",
            "0922111111",
            "0901234567",
            "0901234567",
            "0999999999",
            "0999999999",
        ],
        "order_date": [
            pd.to_datetime("2018-03-01"),
            pd.to_datetime("2018-07-01"),
            pd.to_datetime("2019-08-01"),
            pd.to_datetime("2019-07-01"),
            pd.to_datetime("2018-04-01"),
            pd.to_datetime("2019-05-11"),
            pd.to_datetime("2018-08-01"),
            pd.to_datetime("2019-05-21"),
        ],
    }
    expected_result = ["0922111111"]

    result = get_continuing_buying_for_the_past_two_years(
        pd.DataFrame(data), assess_date=pd.to_datetime("2020-01-03"))

    assert expected_result == result
