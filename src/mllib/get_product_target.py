import pandas as pd

from mllib.data_extraction import ExtractDataForTraining
from mllib.ml_utils.utils import (
    combine_product_target_and_category_for_P03_P04,
    get_p02_training_target,
    get_product_df,
)
from utils.baseclass import PSIBase


class GetProductTargetForP02(ExtractDataForTraining):
    def get_training_target(self) -> pd.DataFrame:
        """取得 P02 訓練目標產品資訊."""
        training_target_info = get_p02_training_target()
        return training_target_info


    def get_output_df(self, sales_df: pd.DataFrame, training_info: pd.DataFrame, category_df: pd.DataFrame) -> pd.DataFrame:

        product_df = get_product_df(
            sales_df=sales_df,
            training_info=training_info,
        )
        category_df = category_df.rename(columns={"sku": "product_id_combo"})
        output_df = product_df.merge(category_df, how="left", on="product_id_combo")
        return output_df


class GetProductTargetForP03P04(ExtractDataForTraining):


    def __init__(self, department_code: str) -> None:
        error_msg = "department_code must be P04, P03"
        if department_code in ["P04", "P03"]:
            self.dep_code = int(department_code[-1:])
        else:
            raise ValueError(error_msg)


    def get_training_target(self) -> pd.DataFrame:
        """取得 P03 P04 訓練目標產品資訊."""
        sh = PSIBase(department=self.dep_code).sh_setting
        training_target_info = (
            sh.worksheet_by_title("品號資料")
            .get_as_df(numerize=False)
            .assign(product_id=lambda df: df["自訂品號"].str.split("/"))
            .rename(
                columns={
                    "自訂品號": "product_id_combo",
                    "自訂品名": "product_name",
                    "品牌": "brand",
                    "product_id": "sku",
                },
            )
        )
        return training_target_info


    def get_output_df(self, sales_df: pd.DataFrame, training_info: pd.DataFrame, category_df: pd.DataFrame) -> pd.DataFrame:

        product_df = get_product_df(
            sales_df=sales_df,
            training_info=training_info,
        )
        product_attr = combine_product_target_and_category_for_P03_P04(
            training_info=training_info,
            categories_df=category_df.drop("brand", axis=1),
        )
        output_df = product_df.merge(product_attr, how="left", on="product_id_combo")
        return output_df


if __name__ == "__main__":
    training_target = GetProductTargetForP02()
    training_info = training_target.get_training_target()
