import logging

import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient
from tqdm import tqdm

from mllib.data_engineering import set_training_weight
from mllib.get_product_target import GetProductTargetForP02, GetProductTargetForP03P04
from utils.baseclass import BomHandler

logging.basicConfig(level=logging.INFO)

class  TransformDataForTraining:
    """將 P02、P03、P04 部門的銷售、產品、分類數據轉換為訓練數據."""

    def __get_product_target_class(self, department_code: str) -> object:
        """根據部門代碼返回對應的 GetProductTarget 對象."""
        if department_code == "P02":
            return GetProductTargetForP02()
        else:
            return GetProductTargetForP03P04(department_code)


    def __init__(self, department_code: str) -> None:
        """初始化 TransformDataForTraining 類別."""
        self.department_code = department_code
        self.data_extraction = self.__get_product_target_class(department_code)


    def get_transformed_training_data(
            self,
            start_date: str,
            end_date: str,
            bigquery_client: BigQueryClient,
    ) -> pd.DataFrame:
        """獲取轉換後的訓練數據."""
        training_info = self.data_extraction.get_training_target()
        category_df = self.data_extraction.get_product_categories(bigquery_client)
        sales_df = self.data_extraction.get_sales_data(
            start_date=start_date,
            end_date=end_date,
            department_code=self.department_code,
            bigquery_client=bigquery_client,
        )
        output_df = self.data_extraction.get_output_df(
            sales_df=sales_df,
            training_info=training_info,
            category_df=category_df,
        )

        date_range = pd.date_range(output_df["date"].min(), output_df["date"].max(), freq="MS")
        realized_sales_months = sales_df.groupby("month_begin_date")
        for month in tqdm(date_range):
            logging.info("BOM 計算月份: %s", str(month.date()))
            # 篩選月資料
            current_month_sales_df = (realized_sales_months.get_group(str(month.date())))
            # 輸入銷貨數量
            bom_instance = BomHandler(
                current_month_sales_df[["sku", "net_sales_quantity"]].rename(columns={"net_sales_quantity": "quantity"}),
                bigquery_client=bigquery_client,
            )
            output_df_month = output_df[output_df["date"] == month]
            product_month_quantity = output_df_month["product_id_combo"].map(bom_instance.get_total_quantity)
            output_df.loc[product_month_quantity.index, "sales"] = product_month_quantity.values
        # 根據資料時間設定權重
        transformed_df = set_training_weight(output_df)
        return transformed_df


if __name__ == "__main__":
    from utils.gcp.client import get_bigquery_client
    bigquery_client = get_bigquery_client()
    trans_model = TransformDataForTraining(department_code="P02")
    result = trans_model.get_transformed_training_data(
        start_date="2023-01-01",
        end_date="2023-03-31",
        bigquery_client=bigquery_client,
    )
