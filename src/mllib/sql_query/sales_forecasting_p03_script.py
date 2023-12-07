

def create_p03_model_testing():
    """Create src_ds.ds_p03_model_testing."""
    bq_query = """
        CREATE OR REPLACE TABLE src_ds.ds_p03_model_testing (
            month_version STRING OPTIONS(description="模型版本"),
            dep_code STRING OPTIONS(description="部門代號"),
            brand STRING OPTIONS(description="品牌"),
            product_category_1 STRING OPTIONS(description="產品大類"),
            product_category_2 STRING OPTIONS(description="產品中類"),
            product_category_3 STRING OPTIONS(description="產品小類"),
            product_id_combo STRING OPTIONS(description="產品名稱組合"),
            product_name STRING OPTIONS(description="品名"),
            date DATETIME OPTIONS(description="日期"),
            predicted_on_date DATETIME OPTIONS(description="預測日期"),
            M INT64 OPTIONS(description="執行當月和預測月份的差"),
            sales FLOAT64 OPTIONS(description="真實銷售量"),
            sales_model FLOAT64 OPTIONS(description="模型預測銷售量"),
            sales_agent INT64 OPTIONS(description="業務預估銷售量"),
            less_likely_lb FLOAT64 OPTIONS(description="寬鬆信心區間下界"),
            likely_lb FLOAT64 OPTIONS(description="嚴謹信心區間下界"),
            likely_ub FLOAT64 OPTIONS(description="嚴謹信心區間上界"),
            less_likely_ub FLOAT64 OPTIONS(description="寬鬆信心區間上界"),
            positive_ind_likely INT64 OPTIONS(description="模型預測落在嚴謹區間內"),
            positive_ind_less_likely INT64 OPTIONS(description="模型預測落在寬鬆區間內"),
        )
    """
    return bq_query


def create_p03_model_predict():
    """Create src_ds.ds_p03_model_predict."""
    bq_query = """
        CREATE OR REPLACE TABLE src_ds.ds_p03_model_predict (
            month_version STRING OPTIONS(description="模型版本"),
            dep_code STRING OPTIONS(description="部門代號"),
            brand STRING OPTIONS(description="品牌"),
            product_category_1 STRING OPTIONS(description="產品大類"),
            product_category_2 STRING OPTIONS(description="產品中類"),
            product_category_3 STRING OPTIONS(description="產品小類"),
            product_id_combo STRING OPTIONS(description="產品名稱組合"),
            product_name STRING OPTIONS(description="品名"),
            date DATETIME OPTIONS(description="日期"),
            predicted_on_date DATETIME OPTIONS(description="預測日期"),
            M INT64 OPTIONS(description="執行當月和預測月份的差"),
            sales_model FLOAT64 OPTIONS(description="模型預測銷售量"),
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
            month_version STRING OPTIONS(description="模型版本"),
            product_id_combo STRING OPTIONS(description="產品名稱組合"),
            bound_01_flag STRING OPTIONS(description="參考性"),
            brand STRING OPTIONS(description="品牌"),
            dep_code STRING OPTIONS(description="部門代號"),
        )
    """
    return bq_query
