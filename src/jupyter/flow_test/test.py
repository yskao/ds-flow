
import pandas as pd
from google.cloud.bigquery import Client as BigQueryClient

from utils.gcp.client import get_bigquery_client


def get_product_categories(bigquery_client: BigQueryClient) -> pd.DataFrame:
    """取得 HLH 資料庫中的產品類資訊。包含品牌、品號 (sku)、產品資訊."""
    query_string = """
    SELECT
        brand,
        sku,
        product_category_1,
        product_category_2,
        product_category_3,
        product_type,
        stock_type,
        accounting_type
    FROM
        dim.products
    """
    product_df = bigquery_client.query(query_string).result().to_dataframe()
    category_df = product_df.loc[
        lambda df: (df["accounting_type"] == "商品存貨") & (df["stock_type"] != "贈品"),
        ["brand", "sku", "product_category_1", "product_category_2", "product_category_3"],
    ]
    return category_df

if __name__ == "__main__":
    bigquery_client=get_bigquery_client()
    bigquery_client.query("SELECT * FROM data-warehouse-369301.dim.products").result().to_dataframe()
