

class CylinderSQL:


    def orders_hs_old_sql() -> str:
        orders_hs_old_sql_query = """
            WITH hs_order_id AS (
                SELECT
                    order_number,
                    external_order_number
                FROM
                    (SELECT
                        CONCAT(TH001, TH002) AS order_number,
                        TH074 AS external_order_number,
                        ROW_NUMBER() OVER (PARTITION BY CONCAT(TH001, TH002) ORDER BY TH074) AS rn
                    FROM `ods_ERP.COPTH`) AS Z
                WHERE rn = 1
            )
            SELECT
                'HS-OLD' AS data_source,
                G.TG003 AS order_date,
                IF(D.MD003 IS NULL, H.TH004, D.MD003) AS sku,
                IF(D.MD003 IS NULL, H.TH008 + TH024, (H.TH008 + TH024) * D.MD006) AS sales_quantity,
                H.TH013 AS sales_amount,
                O.external_order_number,
                P.product_category_1,
                G.TG106 AS mobile
            FROM `ods_ERP.COPTG` G --單頭
            LEFT JOIN `ods_ERP.COPTH` H ON G.TG001 = H.TH001 AND G.TG002 = H.TH002 -- 單身
            LEFT JOIN `dim.products` P ON H.TH004 = P.sku --產品資訊
            LEFT JOIN `ods_ERP.BOMMD` D ON P.product_type = '虛擬組合' AND H.TH004 = D.MD001 --拆產品組合
            LEFT JOIN `ods_ERP.COPTJ` J ON H.TH074 = J.TJ059 AND H.TH004 = J.TJ004 --作廢紀錄
            LEFT JOIN hs_order_id O ON CONCAT (RTRIM(G.TG001), RTRIM(G.TG002)) = O.order_number
            WHERE G.TG004 LIKE 'F901%' --舊官網
                AND J.TJ059 IS NULL --若有作廢交易就不納入計算
                AND (P.product_category_id_2 = '4201' --判斷鋼瓶產品中類
                OR P.product_category_1='氣泡水機')
            """
        return orders_hs_old_sql_query


    def orders_hs_91app_sql() -> str:
        orders_hs_91app_sql_query = """
            SELECT
                'HS-91APP' AS data_source,
                G.TG003 AS order_date,
                IF(D.MD003 IS NULL, H.TH004, D.MD003) AS sku,
                IF(D.MD003 IS NULL, H.TH008 + TH024, (H.TH008 + TH024) * D.MD006) AS sales_quantity,
                H.TH013 AS sales_amount,
                I.MI029 AS mobile,
                P.product_name,
                P.product_category_1
            FROM `ods_ERP.COPTG` G --單頭
            LEFT JOIN `ods_ERP.COPTH` H ON G.TG001 = H.TH001 AND G.TG002 = H.TH002 -- 單身
            LEFT JOIN `dim.products` P ON H.TH004 = P.sku --產品資訊
            LEFT JOIN `ods_ERP.BOMMD` D ON P.product_type = '虛擬組合' AND H.TH004 = D.MD001 --拆產品組合
            LEFT JOIN `ods_ERP.COPTJ` J ON H.TH074 = J.TJ059 AND H.TH004 = J.TJ004 --作廢紀錄
            INNER JOIN `ods_ERP.WSCMI` I ON G.TG108 = I.MI001 AND LENGTH(I.MI029) = 10 --會員資訊
            WHERE G.TG004 LIKE 'F909%' --91APP
                AND J.TJ059 IS NULL --若有作廢交易就不納入計算
                AND (P.product_category_id_2 = '4201' --判斷鋼瓶產品中類
                OR P.product_category_1='氣泡水機')
            """
        return orders_hs_91app_sql_query


    def orders_pos_sql() -> str:
        orders_pos_sql_query = """
            SELECT
                'POS' AS data_source,
                V.TV001 AS order_date,
                IF(D.MD003 IS NULL, V.TV010, D.MD003) AS sku,
                IF(D.MD003 IS NULL, V.TV014, V.TV014 * D.MD006) AS sales_quantity,
                V.TV016 AS sales_amount,
                I.MI029 AS mobile,
                P.product_name,
                P.product_category_1
            FROM `ods_ERP.POSTV` V --單頭單身
            LEFT JOIN `dim.products` P ON V.TV010 = P.sku --產品資訊
            LEFT JOIN `ods_ERP.BOMMD` D ON P.product_type = '虛擬組合' AND V.TV010 = D.MD001 --拆產品組合
            LEFT JOIN `ods_ERP.POSTA` A ON V.TV001 = A.TA042 AND V.TV002 = A.TA044 AND V.TV003 = A.TA045 AND V.TV006 = A.TA043 --作廢紀錄
            INNER JOIN `ods_ERP.WSCMI` I ON V.TV008 = I.MI001 AND LENGTH(I.MI029) = 10 --會員資訊
            WHERE V.TV009 IN ('1', '8') --POS 抓銷售、禮券銷售
                AND V.TV016 >= 0 --排除退貨
                AND A.TA001 IS NULL --作廢交易就不納入計算
                AND (P.product_category_id_2 = '4201' --判斷鋼瓶產品中類
                OR P.product_category_1='氣泡水機')
        """
        return orders_pos_sql_query


    def gas_cylinder_points_sql() -> str:
        gas_cylinder_points_sql_query = """
            SELECT
                Phone,
                SUM(Quantity) AS GasCylinder_Point_Cnt
            FROM
                (
                SELECT
                    '百貨專櫃' AS DateSource_Name,
                    TV.TV001 AS Ord_Date,
                    MA1.MA003 AS Brand_Name,
                    MA2.MA003 AS BrandType_Name,
                    MA3.MA003 AS BrandType2_Name,
                    B.MB001 AS Product_Code,
                    B.MB002 AS Product_Name,
                    (TV.TV014 *
                    (
                        SELECT IFNULL(SUM(MD006), 1)
                        FROM `data-warehouse-369301.ods_ERP.BOMMC`
                        LEFT JOIN `data-warehouse-369301.ods_ERP.BOMMD` ON RTRIM(MC001) = RTRIM(MD001)
                        LEFT JOIN `data-warehouse-369301.ods_ERP.INVMB` ON RTRIM(MD003) = RTRIM(MB001)
                        WHERE 1 = 1
                            AND MC001 = B.MB001
                            AND MB111 = '4201'
                    )) AS Quantity,
                    TV.TV016 AS SaleTax_Amt,
                    RTRIM(MI.MI029) AS Phone,
                    RTRIM(MI.MI001) AS SN,
                    TA001 AS Markout_Date,
                    A.TA002 AS Markout_Store_Code,
                    A.TA003 AS Markout_POS_Code,
                    A.TA006 AS Markout_Txn_Code,
                    CASE WHEN A.TA001 <> '' THEN 'Y' ELSE 'N' END AS Markout_YN
                FROM
                    `data-warehouse-369301.ods_ERP.POSTV` AS TV
                    INNER JOIN `data-warehouse-369301.ods_ERP.WSCMI` AS MI ON TV.TV008 = MI.MI001
                    INNER JOIN `data-warehouse-369301.ods_ERP.INVMB` AS B ON TV.TV010 = RTRIM(B.MB001)
                    LEFT OUTER JOIN `data-warehouse-369301.ods_ERP.INVMA` AS MA1 ON B.MB006 = RTRIM(MA1.MA002) AND MA1.MA001 = '2'
                    LEFT OUTER JOIN `data-warehouse-369301.ods_ERP.INVMA` AS MA2 ON B.MB008 = RTRIM(MA2.MA002) AND MA2.MA001 = '4'
                    LEFT OUTER JOIN `data-warehouse-369301.ods_ERP.INVMA` AS MA3 ON B.MB111 = RTRIM(MA3.MA002) AND MA3.MA001 = '5'
                    LEFT JOIN `data-warehouse-369301.ods_ERP.POSTA` AS A ON RTRIM(TV001) = RTRIM(TA042) AND RTRIM(TV002) = RTRIM(TA044) AND RTRIM(TV003) = RTRIM(TA045) AND RTRIM(TV006) = RTRIM(TA043)
                WHERE
                    (MI.MI029 <> '') AND (MI.MI029 IS NOT NULL)
                    AND (TV.TV009 IN ('1', '8'))
                    AND (TV.TV016 > 0)
                    AND LEFT(TV.TV001, 4) = '2023'
                    AND RTRIM(MA3.MA002) = '4201'
                    AND CASE WHEN TA001 <> '' THEN 'Y' ELSE 'N' END <> 'Y'
                    AND TV.TV001 < FORMAT_DATE('%Y%m%d', (DATE_ADD(CURRENT_DATE('+8'), INTERVAL -2 DAY)))
                    AND LENGTH(RTRIM(MI.MI029)) = 10
                UNION ALL
                SELECT
                    'HS官網' AS DateSource_Name,
                    TG003 AS Ord_Date,
                    B.MA003 AS Brand_Name,
                    C.MA003 AS BrandType_Name,
                    T.MA003 AS BrandType2_Name,
                    TH004 AS Product_Code,
                    MB002 AS Product_Name,
                    ((TH008 + TH024) *
                    (
                        SELECT IFNULL(SUM(MD006), 1)
                        FROM `data-warehouse-369301.ods_ERP.BOMMC`
                        LEFT JOIN `data-warehouse-369301.ods_ERP.BOMMD` ON RTRIM(MC001) = RTRIM(MD001)
                        LEFT JOIN `data-warehouse-369301.ods_ERP.INVMB` ON RTRIM(MD003) = RTRIM(MB001)
                        WHERE 1 = 1
                        AND MB111 = '4201'
                    )) AS Quantity,
                    TH013 AS SaleTax_Amt,
                    RTRIM(MI029) AS Phone,
                    RTRIM(MI001) AS SN,
                    TJ001 AS Markout_Date,
                    TJ002 AS Markout_Store_Code,
                    TJ003 AS Markout_POS_Code,
                    TJ059 AS Markout_Txn_Code,
                    CASE WHEN TJ059 <> '' THEN 'Y' ELSE 'N' END AS Markout_YN
                FROM
                    `data-warehouse-369301.ods_ERP.COPTG`
                    LEFT JOIN `data-warehouse-369301.ods_ERP.COPTH` ON TG001 = TH001 AND TG002 = TH002
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMB` ON TH004 = RTRIM(MB001)
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMA` AS B ON MB006 = RTRIM(B.MA002) AND B.MA001 = '2'
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMA` AS C ON MB008 = RTRIM(C.MA002) AND C.MA001 = '4'
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMA` AS T ON MB111 = RTRIM(T.MA002) AND T.MA001 = '5'
                    INNER JOIN `data-warehouse-369301.ods_ERP.WSCMI` ON TG108 = RTRIM(MI001)
                    LEFT JOIN `data-warehouse-369301.ods_ERP.COPTJ` ON TH074 = TJ059 AND TH004 = TJ004
                    LEFT JOIN `data-warehouse-369301.ods_ERP.COPTC` ON TH014 = TC001 AND TH015 = TC002
                WHERE
                    TG108 <> ''
                    AND (MI029 <> '') AND (MI029 IS NOT NULL)
                    AND TH013 > 0
                    AND LEFT(TG003, 4) = '2023'
                    AND (RTRIM(T.MA002) = '4201')
                    AND CASE WHEN TJ059 <> '' THEN 'Y' ELSE 'N' END <> 'Y'
                    AND TG003 < FORMAT_DATE('%Y%m%d', (DATE_ADD(CURRENT_DATE('+8'), INTERVAL -10 DAY)))
                    AND TC003 >= '20230101'
                    AND LENGTH(RTRIM(MI029)) = 10
            ) AS A
            GROUP BY Phone
        """
        return gas_cylinder_points_sql_query


    def products_df_sql() -> str:
        products_df_sql_query = """
            SELECT * FROM dim.products
        """
        return products_df_sql_query


def cdp_soda_stream_sql() -> str:
    query = """
        CREATE OR REPLACE TABLE CDP.DS_SodaStream_Prediction AS
        WITH ds_soda_stream_prediction AS (
            SELECT * FROM data-warehouse-369301.DS.DS_SodaStream_Prediction
            UNION ALL
            SELECT * FROM data-warehouse-369301.DS.DS_SodaStream_Prediction_TestList
        )
        SELECT
            Member_Mobile AS identity_mobile,
            CAST(Assess_Date AS STRING) AS soda_assess_date,
            Member_GasCylindersReward_Point AS soda_gascylindersreward_point,
            CAST(LastPurchase_Datetime AS STRING) AS soda_lastpurchase_datetime,
            Repurchase_Flag AS soda_repurchase_flag,
            Repurchase_Possibility AS soda_repurchase_possibility,
            New_Counter_Code AS soda_member_counter,
            Avg_Duration_Day_Cnt AS soda_avgdurationday_cnt
        FROM ds_soda_stream_prediction
        LEFT JOIN (
        SELECT
            mobile,
            counter
        FROM dim.members
        ) AS dim_members
        ON ds_soda_stream_prediction.Member_Mobile = dim_members.mobile
        LEFT JOIN (
        SELECT
            Counter_Code,
            (
                CASE
                    WHEN ReplaceDyson_Counter_Code='' THEN Counter_Code
                    ELSE ReplaceDyson_Counter_Code
                END
            ) AS New_Counter_Code
            FROM dim.counter
        ) AS dim_counter
        ON dim_members.counter = dim_counter.Counter_Code
        WHERE Assess_Date = CURRENT_DATE("Asia/Taipei")
        AND Repurchase_Possibility IS NOT NULL;
        """
    return query


def create_p03_model_testing():
    """Create DS.ds_p03_model_testing."""
    bq_query = """
        CREATE OR REPLACE TABLE DS.ds_p03_model_testing (
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
    """Create DS.ds_p03_model_predict."""
    bq_query = """
        CREATE OR REPLACE TABLE DS.ds_p03_model_predict (
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
