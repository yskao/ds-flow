# %%
import logging

import numpy as np
from pandas import DataFrame
from pygsheets.worksheet import Worksheet

logging.basicConfig(level=logging.INFO)


def _get_psi_df_raw(wks: Worksheet, value_render: str = "FORMULA") -> DataFrame:
    tmp_df = wks.get_as_df(numerize=False, value_render=value_render)
    column_list = tmp_df.columns.tolist()
    column_list[0] = "品號"
    column_list[1] = "品名"
    column_list[2] = "進銷存"
    column_list[3] = "列名"
    tmp_df.columns = column_list
    tmp_df = tmp_df.iloc[1:]
    tmp_df["品號"] = tmp_df["品號"].astype(str)

    return tmp_df


def _fill_in_product_code(fill_df: DataFrame) -> DataFrame:
    """因應儲存格合併(資料只留在第一格),補齊品號資料."""
    for idx, data in fill_df.iterrows():
        if data["品號"]:
            target_value = data["品號"]
        else:
            fill_df.loc[idx, "品號"] = target_value
    fill_df = fill_df.reset_index(drop=True)

    return fill_df


def get_psi_df(wks: Worksheet, value_render: str = "FORMULA") -> DataFrame:
    """從google sheet 上取得資料(dataframe)."""
    tmp_df = _get_psi_df_raw(wks, value_render)
    output_df = _fill_in_product_code(tmp_df)
    logging.info("取得 %s 的PSI資料", wks.title)

    return output_df


def get_psi_details(wks: Worksheet) -> DataFrame:
    """取得PSI表上的產品明細資料."""
    tmp_df = _get_psi_df_raw(wks)
    divider_index = tmp_df[tmp_df["列名"] == ""].index.min()
    if np.isnan(divider_index):
        pass
    else:
        tmp_df = tmp_df.iloc[: divider_index - 1]
    output_df = _fill_in_product_code(tmp_df)
    logging.info("取得 %s 的產品明細資料", wks.title)

    return output_df


def get_agent_df(wks: Worksheet, value_render="UNFORMATTED_VALUE") -> DataFrame:
    """取得 google sheet 上的銷售資料."""
    psi_sales_df = wks.get_as_df(value_render=value_render)
    column_list = psi_sales_df.columns.tolist()
    column_list[0] = "品號"
    column_list[1] = "品名"
    column_list[2] = "單價"
    psi_sales_df.columns = column_list

    # 篩選資料
    psi_sales_df = psi_sales_df[psi_sales_df["品名"] != ""]
    psi_sales_df = psi_sales_df[psi_sales_df["品名"] != "品名"]
    psi_sales_df = psi_sales_df[psi_sales_df["品號"] != ""]
    logging.info("取得 %s 的銷售資料", wks.title)

    return psi_sales_df


def get_transfer_df(wks: Worksheet) -> DataFrame:
    """取得 google sheet 的調撥資料."""
    psi_transfer_df = wks.get_as_df(numerize=False)
    columns = psi_transfer_df.columns.tolist()
    columns[0] = "品號"
    columns[1] = "品名"
    psi_transfer_df.columns = columns
    psi_transfer_df = psi_transfer_df.iloc[1:]
    logging.info("取得 %s 的調撥資料", wks.title)

    return psi_transfer_df


def insert_new_row(
    wks: Worksheet,
    idx: int,
    product_code: str,
    product_name: str,
    inherit: bool = False,
) -> None:
    """
    複製列資料,改品號、品名,向下插入一列資料
    註: 插入的列資料變更公式文字,濾掉數值.
    """
    current_row = wks.get_row(row=idx, value_render="FORMULA")
    insert_row = []
    # 變更公式、洗掉數值
    for ele in current_row:
        if isinstance(ele, str) and "=" in ele:
            data = ele.replace(str(idx), str(idx + 1))
        elif isinstance(ele, int):
            data = ""
        else:
            data = ele

        insert_row.append(data)

    insert_row[0] = product_code
    insert_row[1] = product_name
    insert_row[2] = ""
    wks.insert_rows(row=idx, values=insert_row, inherit=inherit)
    logging.info("在 %s 插入 %s", idx, product_code)


def get_working_df(worksheet: Worksheet, month_str: str) -> DataFrame:
    working_df = get_psi_df(worksheet)
    working_df.columns = [str(column) for column in working_df.columns]
    return working_df[["品號", "品名", "進銷存", "列名", month_str]].copy()


def update_data_to_google_sheets(worksheet: Worksheet, df: DataFrame, pattern: str, row_offset: int = 1) -> None:
    output_list = df[pattern].tolist()
    start_cell = worksheet.find(pattern=pattern)[0]
    worksheet.update_col(
        index=start_cell.col, values=output_list, row_offset=start_cell.row + row_offset,
    )

    logging.info(
        "Worksheet {worksheet_title} {date_str} 資料更新完成",
        extra={
            "worksheet_title": worksheet.title,
            "date_str": pattern,
        },
    )


def clear_google_sheets_data(
    working_df: DataFrame,
    clear_data_column_name: str,
    index_name: str,
) -> DataFrame:

    items_df = working_df[working_df["列名"] == clear_data_column_name][["品號"]]

    for idx, _ in items_df.iterrows():
        working_df.loc[idx, index_name] = ""

    return working_df
