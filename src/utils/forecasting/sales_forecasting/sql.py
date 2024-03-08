
def create_p03_model_testing():
    """Create src_ds.ds_p03_model_testing."""
    bq_query = """
        CREATE OR REPLACE TABLE src_ds.ds_p03_model_testing (
            estimate_version STRING OPTIONS(description="模型預估版本"),
            dept_code STRING OPTIONS(description="部門代號"),
            brand_id STRING OPTIONS(description="品牌代號"),
            product_custom_id STRING OPTIONS(description="自訂產品代號"),
            product_custom_name STRING OPTIONS(description="自訂產品名稱"),
            estimate_date DATETIME OPTIONS(description="預估日期"),
            estimate_month_gap INT64 OPTIONS(description="執行當月和預測月份的差"),
            net_sale_qty FLOAT64 OPTIONS(description="真實銷售量"),
            net_sale_qty_model FLOAT64 OPTIONS(description="模型預測銷售量"),
            net_sale_qty_agent FLOAT64 OPTIONS(description="業務預估銷售量"),
            less_likely_lb FLOAT64 OPTIONS(description="寬鬆信心區間下界"),
            likely_lb FLOAT64 OPTIONS(description="嚴謹信心區間下界"),
            likely_ub FLOAT64 OPTIONS(description="嚴謹信心區間上界"),
            less_likely_ub FLOAT64 OPTIONS(description="寬鬆信心區間上界"),
        )
    """
    return bq_query


def create_p03_model_predict():
    """Create src_ds.ds_p03_model_predict."""
    bq_query = """
        CREATE OR REPLACE TABLE src_ds.ds_p03_model_predict (
            estimate_version STRING OPTIONS(description="模型預估版本"),
            dept_code STRING OPTIONS(description="部門代號"),
            brand_id STRING OPTIONS(description="品牌代號"),
            product_custom_id STRING OPTIONS(description="自訂產品代號"),
            product_custom_name STRING OPTIONS(description="自訂產品名稱"),
            estimate_date DATETIME OPTIONS(description="預估日期"),
            estimate_month_gap INT64 OPTIONS(description="執行當月和預測月份的差"),
            net_sale_qty_model FLOAT64 OPTIONS(description="模型預測銷售量"),
            less_likely_lb FLOAT64 OPTIONS(description="寬鬆信心區間下界"),
            likely_lb FLOAT64 OPTIONS(description="嚴謹信心區間下界"),
            likely_ub FLOAT64 OPTIONS(description="嚴謹信心區間上界"),
            less_likely_ub FLOAT64 OPTIONS(description="寬鬆信心區間上界"),
        )
    """
    return bq_query


def create_p03_model_referenceable():
    """Create src_ds.ds_p03_model_referenceable."""
    bq_query = """
        CREATE OR REPLACE TABLE src_ds.ds_p03_model_referenceable (
            estimate_version STRING OPTIONS(description="模型預估版本"),
            brand_id STRING OPTIONS(description="品牌代號"),
            dept_code STRING OPTIONS(description="部門代號"),
            product_custom_id STRING OPTIONS(description="自訂產品代號"),
            reference STRING OPTIONS(description="參考性"),
        )
    """
    return bq_query
