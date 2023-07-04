import re
from datetime import date

import pandas as pd
from pandas import DataFrame
from psi.utils.config import DEPARTMENT_SETTINGS_MAP
from pygsheets.spreadsheet import Spreadsheet
from pygsheets.worksheet import Worksheet
from utilities.common_tools import connect_to_mssql
from utilities.google_sheets import get_google_sheet_client


def create_date_df(year: int) -> DataFrame:
    date_list = pd.date_range(
        start=date(year, 1, 1), end=date(year + 1, 6, 1), freq="MS",
    ).date.tolist()

    date_str_list = [
        date.strftime("%Y%m")
        for date in date_list
    ]

    date_df = pd.DataFrame(columns=date_str_list, index=["status"])
    date_df.loc["status"] = "預估"

    return date_df


def get_product_config_df(department: int, year: int) -> DataFrame:
    setting_sheet_url = DEPARTMENT_SETTINGS_MAP[department].setting_sheet_url[year]

    google_sheet_client = get_google_sheet_client()
    setting_sheet = google_sheet_client.open_by_url(setting_sheet_url)

    return setting_sheet.worksheet_by_title("品號資料").get_as_df(numerize=False)


def create_google_sheets(title: str) -> Spreadsheet:
    google_sheet_client = get_google_sheet_client()

    sheet_info_dict = google_sheet_client.sheet.create(title)
    google_sheet = google_sheet_client.open_by_url(sheet_info_dict["spreadsheetUrl"])

    google_sheet.share("hlh.datateam@gmail.com", role="writer", type="user")

    return google_sheet


def insert_gsheets_info(sheet_type: str, sheet: Spreadsheet, department: int, year: int) -> None:
    with connect_to_mssql("hlh_psi") as connection:
        connection.execute(
            """
            INSERT INTO f_gsheets_info (gsheet_type, sh_id, sh_title, sh_url, department, year, name)
            VALUES (%(gsheet_type)s, %(sh_id)s, %(sh_title)s, %(sh_url)s, %(department)s, %(year)s, %(name)s)
            """,
            gsheet_type=sheet_type,
            sh_id=sheet.id,
            sh_title=sheet.title,
            sh_url=sheet.url,
            department=department,
            year=year,
            name=sheet.title.split("_")[1],
        )


def create_product_df(brand_product_config_df: DataFrame, row_name_tuple_list: list[tuple[str, str]]) -> DataFrame:
    columns = ["品號", "品名", "進銷存", "列名"]

    product_df = pd.DataFrame(columns=columns)

    product_data_rows = []
    for _, row in brand_product_config_df.iterrows():
        for psi_type, name in row_name_tuple_list:
            product_data_rows.append({
                "品號": row["自訂品號"],
                "品名": row["自訂品名"],
                "進銷存": psi_type,
                "列名": name,
            })

    return pd.concat([product_df, pd.DataFrame(product_data_rows)])


def merge_product_cells(
    worksheet: Worksheet,
    start: str,
    brand_product_config_df: DataFrame,
    row_count_per_product: int,
) -> None:

    # 'A1' -> ['A', '1']
    split_alpha_and_number_list = re.findall(r"[^\W\d_]+|\d+", start)

    column_index = split_alpha_and_number_list[0]
    row_index = int(split_alpha_and_number_list[-1])

    row_start_index = row_index
    for _ in range(brand_product_config_df.shape[0]):
        row_end_index = row_start_index + row_count_per_product - 1
        start = f"{column_index}{row_start_index}"
        end = f"{column_index}{row_end_index}"
        worksheet.merge_cells(start=start, end=end)

        row_start_index = row_end_index + 1
