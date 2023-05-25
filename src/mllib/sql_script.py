

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
