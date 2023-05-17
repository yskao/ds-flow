

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


    def products_df_sql() -> str:
        products_df_sql_query = """
            SELECT * FROM dim.products
        """
        return products_df_sql_query
