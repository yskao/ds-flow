import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient

from utils.date import gen_month_str_list, get_year_start_and_end
from utils.google_sheets import get_google_sheet_client
from utils.psi.utils.config import DEPARTMENT_SETTINGS_MAP, PSI_ROW_NAME_DICT, TABLE_NAME_DICT

tz = "Asia/Taipei"


class PSIBase:
    def set_psi_attr(self, department: int) -> None:
        if department == 3:
            self.psi_attr = PSI_ROW_NAME_DICT[department]
            keys = list(self.psi_attr.keys())
            self.row_name_p = keys[0]
            self.row_name_yoy = keys[1]
            self.row_name_total_s = keys[2]
            self.row_name_7t = keys[3]
            self.row_name_other_t = keys[4]
            self.row_name_3s = keys[5]
            self.channels = keys[6:12]
            self.row_name_i_deficit = keys[12]
            self.row_name_i_01_daily = keys[13]
            self.row_name_i_01 = keys[14]
            self.row_name_i_other = keys[15]
            self.row_name_i_total = keys[16]
        elif department == 4:
            self.psi_attr = PSI_ROW_NAME_DICT[department]
            keys = list(self.psi_attr.keys())
            self.row_name_p = keys[0]
            self.row_name_yoy = keys[1]
            self.row_name_total_s = keys[2]
            self.row_name_7t = keys[3]
            self.row_name_other_t = keys[4]
            self.row_name_4s = keys[5]
            self.channels = keys[6:12]
            self.row_name_i_deficit = keys[12]
            self.row_name_i_01 = keys[13]
            self.row_name_i_other = keys[14]
            self.row_name_i_total = keys[15]

    def __init__(self, department: int = 3, year: int | None = None) -> None:
        department_settings = DEPARTMENT_SETTINGS_MAP[department]
        self.department = department
        self.department_code = department_settings.department_code
        self.full_department_code = department_settings.full_department_code
        self.department_chinese = department_settings.department_chinese

        # 初始化時間與欄位
        year = year or datetime.now(ZoneInfo(tz)).year
        self.year = year
        self.month_list = gen_month_str_list(*get_year_start_and_end(year))
        psi_all_col_list = self.month_list + [
            str(int(month) + 100) for month in self.month_list[:6]
        ]
        self.psi_all_col_list = psi_all_col_list

        # 輸入gc, sh_setting
        self.gc = get_google_sheet_client()
        self.sh_setting = self.gc.open_by_url(
            department_settings.setting_sheet_url[year],
        )
        self.sh_template = self.gc.open_by_url(
            department_settings.template_sheet_url[year],
        )

        # PSI 列名設定
        self.set_psi_attr(department)

        # 設定表名
        self.table_name = TABLE_NAME_DICT


    def get_gsheet_input(self):
        """
        設定資料輸入
        輸入 google sheet 上的 PSI 設定資料,記錄在self
        品號資料: self.brand_product_info
        業務負責通路資料: self.agent_cust_info
        品牌資料: self.brand_info.
        """
        wks_setting = self.sh_setting.worksheet_by_title("品號資料")
        self.brand_product_info = wks_setting.get_as_df(numerize=False)

        wks_setting = self.sh_setting.worksheet_by_title("業務負責通路資料")
        self.agent_cust_info = wks_setting.get_as_df(numerize=False)

        if self.department == 3:
            wks_setting = self.sh_setting.worksheet_by_title("品牌資料")
            self.brand_info = wks_setting.get_as_df(numerize=False)

            wks_setting = self.sh_setting.worksheet_by_title("沒客代銷售目標")
            self.target_info = wks_setting.get_as_df(numerize=False)
        elif self.department == 4:
            wks_setting = self.sh_setting.worksheet_by_title("沒客代銷售目標")
            self.target_info = wks_setting.get_as_df(numerize=False)

        logging.info("取得 %s 資訊", self.sh_setting.title)


class BomHandler:
    """
    用來處理數量計算。因為相同商品可能出現在不同商品組合中,需要考慮組合商品的數量。
    將元件商品(product_code)數量補上所有組合商品中的元件商品數量。
    使用方法:
    輸入數量資料(定義計算範圍)來實例化,欄位需有: 品號、數量
    get_total_inventory_indiv(product_code) 回傳單一品號所有庫存
    get_total_inventory_combo(combo_product_code) 回傳組合品號所有庫存.
    """

    def __init__(self, quantity_data: pd.DataFrame, bigquery_client: BigQueryClient) -> None:
        """品號: sku, 元件品號: component_sku, 組成用量: component_quantity."""
        sql = """
                SELECT
                    RTRIM(MD001) AS sku,
                    MD003 AS component_sku,
                    MD006 AS component_quantity,
                FROM
                    `ods_ERP.BOMMD` MD
                LEFT JOIN
                    `ods_ERP.INVMB` MB
                ON
                    RTRIM(MB.MB001) = RTRIM(MD003)
                WHERE
                    MB.MB005 = '1001'
                -- AND MB.MB115 = 'T000'
            """
        self.bom_df = bigquery_client.query(sql).result().to_dataframe()
        self.quantity_data = quantity_data[["sku", "quantity"]]

    def get_component_quantity(self, product_code):
        """
        計算元件品號的成品數量
        成品數量: product_quantity.
        """
        product_code_list = product_code.split("/")
        bom_component = self.bom_df.query(f"sku in {product_code_list}")
        bom_component = bom_component[
            ~bom_component["component_sku"].isin(product_code_list)
        ]  # 避免重複計算
        quantity_component = self.quantity_data.rename(columns={"sku": "component_sku"})
        component_df = bom_component.merge(quantity_component, how="left", on="component_sku")
        component_df["product_quantity"] = component_df["quantity"] / component_df["component_quantity"]
        quantity_from_component = component_df["product_quantity"].sum()

        return quantity_from_component

    def get_total_quantity(self, product_code):
        """計算product_code的所有數量(單一品號或組合品號)."""
        if "/" in product_code:
            total_sum = self.get_total_quantity_combo(product_code)
        else:
            total_sum = self.get_total_quantity_indiv(product_code)

        return total_sum

    def get_total_quantity_indiv(
        self, product_code: str, product_code_list: list | None = None,
    ):
        """計算單一品號的所有數量."""
        self.product_code = product_code
        exchange_rate_df = self.get_exchange_rate(product_code_list)
        compute_dict = self.replace_exchange_unit(exchange_rate_df)
        total_quantity = self.calculate_total_quantity(compute_dict)

        return total_quantity

    def get_total_quantity_combo(self, combo_product_code):
        """計算合併品號(以'/'分隔單一品號)的所有數量."""
        product_code_list = combo_product_code.split("/")
        total_quantity_items = 0
        for product_code in product_code_list:
            total_quantity_items += self.get_total_quantity_indiv(
                product_code, product_code_list,
            )

        return total_quantity_items

    def get_exchange_rate(self, product_code_list: list | None):
        """
        整理所有兌換比例並輸出,找出所有上游品號與兌換比例
        輸出格式: dataframe(品號, 元件品號, 組成用量).
        """
        init_df = pd.DataFrame()

        bom_df = self.bom_df[self.bom_df["sku"] != self.product_code].copy()
        if product_code_list:
            bom_df = bom_df[~bom_df["sku"].isin(product_code_list)].copy()

        bom_package = bom_df[bom_df["component_sku"] == self.product_code]
        concat_df = pd.concat([init_df, bom_package], axis=0)
        concat_len = len(concat_df)

        if concat_len == 0:
            return concat_df

        new_len = concat_len
        old_len = 0
        while new_len > old_len:
            old_len = len(concat_df)
            for _, data in concat_df.iterrows():
                if data["sku"] in concat_df["component_sku"].unique():
                    continue

                bom_package_of_package = bom_df[bom_df["component_sku"] == data["sku"]]
                concat_df = pd.concat([concat_df, bom_package_of_package], axis=0)

            new_len = len(concat_df)

        exchange_rate_df = concat_df
        return exchange_rate_df

    def replace_exchange_unit(self, exchange_rate_df):
        """
        將所有兌換單位(包含上游包裝品號)換成輸入品號,輸出置換後兌換比例dict
        輸出格式: {品號:兌換比例}.
        """
        compute_dict = {}
        compute_dict[self.product_code] = 1

        for _, data in exchange_rate_df.iterrows():
            comp_product = data["component_sku"]
            if comp_product == self.product_code:
                compute_dict[data["sku"]] = int(data["component_quantity"])
            else:
                multiplexer_list = []
                while comp_product != self.product_code:
                    multiplier = compute_dict[comp_product]
                    multiplexer_list.append(multiplier)
                    comp_product = exchange_rate_df.loc[
                        exchange_rate_df["sku"] == comp_product, "component_sku",
                    ].values[0]
                output_multiplexer = int(data["component_quantity"])
                for multiplier in multiplexer_list:
                    output_multiplexer = output_multiplexer * multiplier
                compute_dict[data["sku"]] = output_multiplexer

        return compute_dict

    def calculate_total_quantity(self, compute_dict):
        """計算真實所有數量."""
        quantity = self.quantity_data[
            self.quantity_data["sku"].isin(compute_dict)
        ].copy()
        if len(quantity) != 0:
            for key, value in compute_dict.items():
                quantity.loc[quantity["sku"] == key, "exchange_rate"] = value
            total_quantity = sum(quantity["quantity"] * quantity["exchange_rate"])
        else:
            total_quantity = 0

        return total_quantity


def items_sum(items, merge_df):
    """
    (items 品號不做split處理)
    items為只有品號的dataframe
    merge_df為只有品號與待加欄位的dataframe.
    """
    items_in_function = items.copy()

    merge_to_compute = items_in_function.merge(merge_df, how="left", on="sku")
    merge_sum = merge_to_compute.groupby(by="sku").sum().reset_index()

    output = items[["sku"]].merge(merge_sum, how="left", on="sku")
    output = output.set_index(items.index)
    return output
