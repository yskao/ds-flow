import numpy as np
import pandas as pd


def cal_cylinder_purchase_qty(orders_df: pd.DataFrame, mapping_combo_qty: dict, assess_date: pd.Timestamp) -> pd.DataFrame:
    """計算鋼瓶購買數量，若有購買組合，會解 bom"""
    orders_tmp_df = orders_df.copy().assign(
        purchase_qty = np.where(orders_df["product_category_2"]=="鋼瓶", orders_df["sales_quantity"], 0)
    )
    tmp_df = orders_tmp_df.query("product_name.str.contains('鋼瓶', na=False)").copy()
    for combo in mapping_combo_qty.keys():
        tmp_df.loc[lambda df: df["product_name"].str.contains(combo), "sales_quantity"] = mapping_combo_qty.get(combo)
        tmp_df.loc[lambda df: df["product_name"].str.contains(combo), "purchase_qty"] = mapping_combo_qty.get(combo)
    orders_tmp_df.loc[tmp_df.index] = tmp_df
    result = (
        orders_tmp_df
        .query(f"order_date >= '{assess_date.year}'")
        .groupby("mobile", as_index=False)
        ["purchase_qty"].sum()
    )
    return result
