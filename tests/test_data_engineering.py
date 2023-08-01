import pandas as pd
from mllib.data_engineering import (
    get_continuing_buying_weights,
    prepare_predict_table_to_sql,
)


def test_prepare_predict_table_to_sql(p03_training_info, p03_predict_result):
    prepare_predict_table_to_sql(
        predict_df=p03_predict_result,
        product_data_info=p03_training_info,
        predicted_on_date="2023-06-01",
        department_code="P03",
    )


def test_get_continuing_buying_weights():

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
    expected_result = pd.DataFrame(
        data={2018: [1, 1], 2019: [2, 0], "seasonal_weight": [0.1, 0.0]},
        index=["0922111111", "0999999999"],
)
    result = get_continuing_buying_weights(
        pd.DataFrame(data),
        assess_date=pd.to_datetime("2020-01-03"),
        summer_period=[6,7,8,9],
        past_year=2,
    )
    assert all(expected_result == result)

    # 三個號碼只有在 2019 有紀錄,所以權重應為 0
    expected_result = pd.DataFrame(
        data={2019: [1, 2, 1], "seasonal_weight": [0.0, 0.0, 0.0]},
        index=["0901234567", "0922111111", "0999999999"],
    )
    result = get_continuing_buying_weights(
        pd.DataFrame(data),
        assess_date=pd.to_datetime("2021-01-03"),
        summer_period=[5,6,7,8,9,10],
        past_year=2,
    )
    assert all(expected_result == result)

    # 三個號碼在 2021、2020 都沒有資料,應為空值
    expected_result = pd.DataFrame(columns=["seasonal_weight"])
    result = get_continuing_buying_weights(
        pd.DataFrame(data),
        assess_date=pd.to_datetime("2022-01-03"),
        summer_period=[5,6,7,8,9,10],
        past_year=2,
    )
    assert all(expected_result == result)

    # past year = 5, 四個號碼在過去都有紀錄
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
            "0987654321",
            "0987654321",
            "0987654321",
            "0987654321",
            "0987654321",
            "0987654321",
            "0987654321",
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
            pd.to_datetime("2017-06-01"),
            pd.to_datetime("2018-05-02"),
            pd.to_datetime("2019-12-05"),
            pd.to_datetime("2020-04-30"),
            pd.to_datetime("2021-07-07"),
            pd.to_datetime("2022-07-04"),
            pd.to_datetime("2023-06-28"),
        ],
    }
    expected_result = pd.DataFrame(
        data={
            2018: [0,1,1,1],
            2019: [1,2,0,1],
            2021: [0,0,1,0],
            2022: [0,0,1,0],
            "seasonal_weight": [0.0, 0.1, 0.2, 0.1],
        },
        index=["0901234567", "0922111111", "0987654321", "0999999999"],
    )
    result = get_continuing_buying_weights(
        pd.DataFrame(data),
        assess_date=pd.to_datetime("2023-01-03"),
        summer_period=[5,6,7,8,9,10],
        past_year=5,
    )
    assert all(expected_result == result)
