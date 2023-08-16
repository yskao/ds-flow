"""
TODO<Sam> soda_stream_repurchase_script.py - cdp_soda_stream_sql_v1  and cdp_soda_stream_campaign_sql
https://app.asana.com/0/1202942986616169/1205265288881501/f.
"""

class CylinderSQL:
    """soda stream bq script."""

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
            INNER JOIN `ods_ERP.WSCMI` I ON RTRIM(G.TG108) = RTRIM(I.MI001) AND LENGTH(I.MI029) = 10 --會員資訊
            WHERE G.TG004 LIKE 'F909%' --91APP
                AND J.TJ059 IS NULL --若有作廢交易就不納入計算
                AND (P.product_category_id_2 = '4201' --判斷鋼瓶產品中類
                AND G.TG023 = "Y"
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
                V.TV021 AS remark,
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
                    '百貨專櫃' AS DateSource_Name, -- 資料來源
                    TV.TV001 AS Ord_Date, -- 日期
                    MA1.MA003 AS Brand_Name, -- 品牌
                    MA2.MA003 AS BrandType_Name, -- 類別
                    MA3.MA003 AS BrandType2_Name, -- 類別2
                    B.MB001 AS Product_Code, -- 料號
                    B.MB002 AS Product_Name, -- 品名
                    (TV.TV014 *
                        (
                        SELECT IFNULL(SUM(MD006), 1) -- 組成用量
                        FROM `data-warehouse-369301.ods_ERP.BOMMC`
                        LEFT JOIN `data-warehouse-369301.ods_ERP.BOMMD` ON RTRIM(MC001) = RTRIM(MD001)
                        LEFT JOIN `data-warehouse-369301.ods_ERP.INVMB` ON RTRIM(MD003) = RTRIM(MB001)
                        WHERE 1 = 1
                            AND MC001 = B.MB001
                            AND MB111 = '4201'
                        )
                    ) AS Quantity, -- 數量
                    TV.TV016 AS SaleTax_Amt, -- 消費金額
                    RTRIM(MI.MI029) AS Phone,
                    RTRIM(MI.MI001) AS SN,
                    TA001 AS Markout_Date, -- 作廢營業日期
                    A.TA002 AS Markout_Store_Code, -- 作廢店號
                    A.TA003 AS Markout_POS_Code, -- 作廢機號
                    A.TA006 AS Markout_Txn_Code, -- 作廢交易序號
                    CASE WHEN A.TA001 <> '' THEN 'Y' ELSE 'N' END Markout_YN -- 是否有作廢
                FROM
                    `data-warehouse-369301.ods_ERP.POSTV` AS TV
                    INNER JOIN `data-warehouse-369301.ods_ERP.WSCMI` AS MI ON TV.TV008 = MI.MI001
                    INNER JOIN `data-warehouse-369301.ods_ERP.INVMB` AS B ON TV.TV010 = RTRIM(B.MB001)
                    LEFT OUTER JOIN `data-warehouse-369301.ods_ERP.INVMA` AS MA1 ON B.MB006 = RTRIM(MA1.MA002) AND MA1.MA001 = '2'
                    LEFT OUTER JOIN `data-warehouse-369301.ods_ERP.INVMA` AS MA2 ON B.MB008 = RTRIM(MA2.MA002) AND MA2.MA001 = '4'
                    LEFT OUTER JOIN `data-warehouse-369301.ods_ERP.INVMA` AS MA3 ON B.MB111 = RTRIM(MA3.MA002) AND MA3.MA001 = '5'
                    LEFT JOIN `data-warehouse-369301.ods_ERP.POSTA` AS A ON RTRIM(TV001) = RTRIM(TA042) AND RTRIM(TV002) = RTRIM(TA044) AND RTRIM(TV003) = RTRIM(TA045) AND RTRIM(TV006) = RTRIM(TA043) -- 202211_判斷有沒有作廢交易
                WHERE
                    (MI.MI029 <> '') AND (MI.MI029 IS NOT NULL) -- 行動電話
                    AND (TV.TV009 IN ('1', '8')) -- 抓銷售、禮券銷售
                    AND (TV.TV016 > 0) -- 交易金額 > 0
                    AND LEFT(TV.TV001, 4) = '2023' -- 年度
                    AND RTRIM(MA3.MA002) = '4201' -- 判斷鋼瓶產品中類
                    AND CASE WHEN TA001 <> '' THEN 'Y' ELSE 'N' END <> 'Y' -- 202211_若有作廢交易就不納入計算
                    AND TV.TV001 < FORMAT_DATE('%Y%m%d', (DATE_ADD(CURRENT_DATE('+8'), INTERVAL -2 DAY)))
                    AND LENGTH(RTRIM(MI.MI029)) = 10
                UNION ALL
                SELECT
                    'HS官網' AS DateSource_Name, -- 資料來源
                    TG003 AS Ord_Date, -- 日期
                    B.MA003 AS Brand_Name, -- 品牌
                    C.MA003 AS BrandType_Name, -- 類別
                    T.MA003 AS BrandType2_Name, -- 類別2
                    TH004 AS Product_Code, -- 料號
                    MB002 AS Product_Name, -- 品名
                    ((TH008 + TH024) *
                        (
                        SELECT IFNULL(SUM(MD006), 1) -- 組成用量
                        FROM `data-warehouse-369301.ods_ERP.BOMMC`
                        LEFT JOIN `data-warehouse-369301.ods_ERP.BOMMD` ON RTRIM(MC001) = RTRIM(MD001)
                        LEFT JOIN `data-warehouse-369301.ods_ERP.INVMB` ON RTRIM(MD003) = RTRIM(MB001)
                        WHERE 1 = 1
                            AND RTRIM(MC001) = RTRIM(TH004)
                            AND MB111 = '4201'
                        )
                    ) AS Quantity, -- 數量
                    TH013 AS SaleTax_Amt, -- 消費金額
                    RTRIM(MI029) AS Phone,
                    RTRIM(MI001) AS SN,
                    TJ001 AS Markout_Date, -- 作廢營業日期
                    TJ002 AS Markout_Store_Code, -- 作廢店號
                    TJ003 AS Markout_POS_Code,
                    TJ059 AS Markout_Txn_Code, -- 作廢交易序號
                    CASE WHEN TJ059 <> '' THEN 'Y' ELSE 'N' END Markout_YN -- 是否有作廢
                FROM
                    `data-warehouse-369301.ods_ERP.COPTG`
                    LEFT JOIN `data-warehouse-369301.ods_ERP.COPTH` ON TG001 = TH001 AND TG002 = TH002
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMB` ON TH004 = RTRIM(MB001)
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMA` AS B ON MB006 = RTRIM(B.MA002) AND B.MA001 = '2'
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMA` AS C ON MB008 = RTRIM(C.MA002) AND C.MA001 = '4'
                    LEFT JOIN `data-warehouse-369301.ods_ERP.INVMA` AS T ON MB111 = RTRIM(T.MA002) AND T.MA001 = '5'
                    INNER JOIN `data-warehouse-369301.ods_ERP.WSCMI` ON TG108 = RTRIM(MI001)
                    LEFT JOIN `data-warehouse-369301.ods_ERP.COPTJ` ON TH074 = TJ059 AND TH004 = TJ004 -- 判斷有沒有作廢交易
                    LEFT JOIN `data-warehouse-369301.ods_ERP.COPTC` ON TH014 = TC001 AND TH015 = TC002 -- 撈取來源訂單
                WHERE
                    TG108 <> '' -- 單據有紀錄會員編號
                    AND (MI029 <> '') AND (MI029 IS NOT NULL) -- 行動電話
                    AND TH013 > 0 -- 交易金額 > 0
                    AND LEFT(TG003, 4) = '2023' -- 年度
                    AND (RTRIM(T.MA002) = '4201') -- 判斷鋼瓶產品中類
                    AND CASE WHEN TJ059 <> '' THEN 'Y' ELSE 'N' END <> 'Y' -- 若有作廢交易就不納入計算
                    AND TG003 < FORMAT_DATE('%Y%m%d', (DATE_ADD(CURRENT_DATE('+8'), INTERVAL -10 DAY))) -- 交易10內的不納入計算
                    AND TC003 >= '20230101' -- 機制2023年的下訂資訊才開始計算
                    AND LENGTH(RTRIM(MI029)) = 10
                ) AS A
            GROUP BY
                Phone
            ORDER BY
                Phone
        """
        return gas_cylinder_points_sql_query


    def mart_ds_sodastream_campaign_last2y_sql() -> str:
        return """SELECT * FROM data-warehouse-369301.dbt_mart_ds.mart_ds_sodastream_campaign_last2y_v"""


    def products_df_sql() -> str:
        products_df_sql_query = """
            SELECT * FROM dim.products
        """
        return products_df_sql_query


def cdp_soda_stream_sql() -> str:
    query = """
        CREATE OR REPLACE TABLE CDP.DS_SodaStream_Prediction AS
        WITH source AS (
            SELECT * FROM `data-warehouse-369301.DS.DS_SodaStream_Prediction`
            UNION ALL
            SELECT * FROM `data-warehouse-369301.DS.DS_SodaStream_Prediction_TestList`
        )

        , prediction AS (
            SELECT
                Member_Mobile AS identity_mobile,
                CAST(Assess_Date AS STRING) AS soda_assess_date,
                Member_GasCylindersReward_Point AS soda_gascylindersreward_point,
                CAST(LastPurchase_Datetime AS STRING) AS soda_lastpurchase_datetime,
                Repurchase_Flag AS soda_repurchase_flag,
                Repurchase_Possibility AS soda_repurchase_possibility,
                Avg_Duration_Day_Cnt AS soda_avgdurationday_cnt
            FROM source
            WHERE Assess_Date = CURRENT_DATE("Asia/Taipei")
                AND Repurchase_Possibility IS NOT NULL
        )

        -- member
        , dim_member AS (
            SELECT
                mobile,
                counter
            FROM `dim.members`
        )

        -- counter
        , dim_counter AS (
            SELECT
                Counter_Code,
                IF(ReplaceDyson_Counter_Code="", Counter_Code, ReplaceDyson_Counter_Code) AS soda_member_counter,
            FROM `dim.counter`
        )

        , cdp_prediction AS (
            SELECT * FROM prediction p
            LEFT JOIN dim_member m ON p.identity_mobile = m.mobile
            LEFT JOIN dim_counter c ON m.counter = c.Counter_Code

        )

        SELECT
            identity_mobile,
            soda_assess_date,
            soda_gascylindersreward_point,
            soda_lastpurchase_datetime,
            soda_repurchase_flag,
            soda_repurchase_possibility,
            soda_member_counter,
            soda_avgdurationday_cnt
        FROM cdp_prediction
        """
    return query


def cdp_soda_stream_sql_v1() -> str:
    query = """
        CREATE OR REPLACE TABLE DS.CDP_DS_SodaStream_Prediction_v1 AS (
            WITH source AS (
                SELECT * FROM `data-warehouse-369301.DS.DS_SodaStream_Prediction_v1`
                UNION ALL
                SELECT * FROM `data-warehouse-369301.DS.DS_SodaStream_Prediction_TestList_v1`
            )

            , prediction AS (
                SELECT
                    Member_Mobile AS identity_mobile,
                    CAST(Assess_Date AS STRING) AS soda_assess_date,
                    Member_GasCylindersReward_Point AS soda_gascylindersreward_point,
                    CAST(LastPurchase_Datetime AS STRING) AS soda_lastpurchase_datetime,
                    Repurchase_Flag AS soda_repurchase_flag,
                    Repurchase_Possibility AS soda_repurchase_possibility,
                    Avg_Duration_Day_Cnt AS soda_avgdurationday_cnt
                FROM source
                WHERE Assess_Date = CURRENT_DATE("Asia/Taipei")
                    AND Repurchase_Possibility IS NOT NULL
            )

            -- member
            , dim_member AS (
                SELECT
                    mobile,
                    counter
                FROM `dim.members`
            )

            -- counter
            , dim_counter AS (
                SELECT
                    Counter_Code,
                    IF(ReplaceDyson_Counter_Code="", Counter_Code, ReplaceDyson_Counter_Code) AS soda_member_counter,
                FROM `dim.counter`
            )

            , cdp_prediction AS (
                SELECT * FROM prediction p
                LEFT JOIN dim_member m ON p.identity_mobile = m.mobile
                LEFT JOIN dim_counter c ON m.counter = c.Counter_Code

            )

            SELECT
                identity_mobile,
                soda_assess_date,
                soda_gascylindersreward_point,
                soda_lastpurchase_datetime,
                soda_repurchase_flag,
                soda_repurchase_possibility,
                soda_member_counter,
                soda_avgdurationday_cnt
            FROM cdp_prediction
        )
        """
    return query


def cdp_soda_stream_campaign_sql() -> str:

    query = """
        CREATE OR REPLACE TABLE CDP.DS_SodaStream_Campaign_v1 AS (
            WITH source AS (
                SELECT * FROM `data-warehouse-369301.DS.DS_SodaStream_Prediction_v1`
                UNION ALL
                SELECT * FROM `data-warehouse-369301.DS.DS_SodaStream_Prediction_TestList_v1`
            )

            -- prediction
            , campaign AS (
                SELECT
                    Member_Mobile AS identity_mobile,
                    CAST(TY_Campaign_Year_ID AS STRING) AS soda_ty_campaign_year_id,
                    CAST(TY_Point_All_Cnt AS INTEGER) AS soda_ty_point_all_cnt,
                    CAST(TY_Point_ToCoupon_Cnt AS INTEGER) AS soda_ty_point_tocoupon_cnt,
                    CAST(TY_Point_Used_Cnt AS INTEGER) AS soda_ty_point_used_cnt,
                    CAST(TY_Point_Unused_Cnt AS INTEGER) AS soda_ty_point_unused_cnt,
                    CAST(TY_Point_UnusedWOCoupon_Cnt AS INTEGER) AS soda_ty_point_unusedwocoupon_cnt,
                    CAST(TY_Coupon_Sent_Cnt AS INTEGER) AS soda_ty_coupon_sent_cnt,
                    CAST(TY_Coupon_Used_Cnt AS INTEGER) AS soda_ty_coupon_used_cnt,
                    CAST(TY_Coupon_Unused_Cnt AS INTEGER) AS soda_ty_coupon_unused_cnt,
                    CAST(TY_Coupon_Sent_ID AS STRING) AS soda_ty_coupon_sent_id,
                    CAST(TY_Coupon_Used_ID AS STRING) AS soda_ty_coupon_used_id,
                    CAST(TY_Coupon_Unused_ID AS STRING) AS soda_ty_coupon_unused_id,
                    CAST(TY_Coupon_Exp_Date AS STRING) AS soda_ty_coupon_exp_date,
                    CAST(LY_Campaign_Year_ID AS STRING) AS soda_ly_campaign_Year_id,
                    CAST(LY_Point_All_Cnt AS INTEGER) AS soda_ly_point_all_cnt,
                    CAST(LY_Point_ToCoupon_Cnt AS INTEGER) AS soda_ly_point_tocoupon_cnt,
                    CAST(LY_Point_Used_Cnt AS INTEGER) AS soda_ly_point_used_cnt,
                    CAST(LY_Point_Unused_Cnt AS INTEGER) AS soda_ly_point_unused_cnt,
                    CAST(LY_Point_UnusedWOCoupon_Cnt AS INTEGER) AS soda_ly_point_unusedwocoupon_cnt,
                    CAST(LY_Coupon_Sent_Cnt AS INTEGER) AS soda_ly_coupon_sent_cnt,
                    CAST(LY_Coupon_Used_Cnt AS INTEGER) AS soda_ly_coupon_used_cnt,
                    CAST(LY_Coupon_Unused_Cnt AS INTEGER) AS soda_ly_coupon_unused_cnt,
                    CAST(LY_Coupon_Sent_ID AS STRING) AS soda_ly_coupon_sent_id,
                    CAST(LY_Coupon_Used_ID AS STRING) AS soda_ly_coupon_used_id,
                    CAST(LY_Coupon_Unused_ID AS STRING) AS soda_ly_coupon_unused_id,
                    CAST(LY_Coupon_Exp_Date AS STRING) AS soda_ly_coupon_exp_date
                FROM source
                WHERE Assess_Date = CURRENT_DATE("Asia/Taipei")
                    AND Repurchase_Possibility IS NOT NULL
            )

            SELECT * FROM campaign
        )
    """
    return query
