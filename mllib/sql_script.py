

class CylinderSQL:


    def orders_hs_old_sql() -> str:
        orders_hs_old_sql_query = """
            WITH hs_order_id AS (
                SELECT
                    訂單編號,
                    外部單號 FROM (
                SELECT
                    CONCAT(TH001, TH002) AS 訂單編號,
                    TH074 AS 外部單號,
                    ROW_NUMBER() OVER (PARTITION BY CONCAT(TH001, TH002) ORDER BY TH074) AS rn
                FROM COPTH) Z
                WHERE rn = 1
            )

            SELECT
                'HS-OLD' AS 資料來源,
                G.TG003 AS 日期,
                IIF(D.MD003 IS NULL, H.TH004, D.MD003) AS 品號,
                IIF(D.MD003 IS NULL, H.TH008 + TH024, (H.TH008 + TH024) * D.MD006) AS 數量,
                TH013 AS 消費金額,
                O.外部單號,
                產品大類,
                TG106 AS 手機
            FROM COPTG G --單頭
            LEFT JOIN COPTH H ON G.TG001 = H.TH001 AND G.TG002 = H.TH002 -- 單身
            LEFT JOIN HLH_DW.dbo.dim_products P ON H.TH004 = P.品號 COLLATE Chinese_Taiwan_Stroke_CI_AS --產品資訊
            LEFT JOIN BOMMD D ON P.產品性質 = '虛擬組合' AND H.TH004 = D.MD001 --拆產品組合
            LEFT JOIN COPTJ J ON H.TH074 = J.TJ059 AND H.TH004 = J.TJ004 --作廢紀錄
            LEFT JOIN hs_order_id O ON CONCAT (RTRIM(G.TG001), RTRIM(G.TG002)) = O.訂單編號
            WHERE G.TG004 LIKE 'F901%' --舊官網
                AND J.TJ059 IS NULL --若有作廢交易就不納入計算
                AND (P.產品中類代號 = '4201' --判斷鋼瓶產品中類
                OR P.產品大類='氣泡水機')
            """
        return orders_hs_old_sql_query


    def orders_hs_91app_sql() -> str:
        orders_hs_91app_sql_query = """
            SELECT
                'HS-91APP' AS 資料來源,
                G.TG003 AS 日期,
                IIF(D.MD003 IS NULL, H.TH004, D.MD003) AS 品號,
                IIF(D.MD003 IS NULL, H.TH008 + TH024, (H.TH008 + TH024) * D.MD006) AS 數量,
                TH013 AS 消費金額,
                I.MI029 AS 手機,
                品名,
                產品大類
            FROM COPTG G --單頭
            LEFT JOIN COPTH H ON G.TG001 = H.TH001 AND G.TG002 = H.TH002 -- 單身
            LEFT JOIN HLH_DW.dbo.dim_products P ON H.TH004 = P.品號 COLLATE Chinese_Taiwan_Stroke_CI_AS --產品資訊
            LEFT JOIN BOMMD D ON P.產品性質 = '虛擬組合' AND H.TH004 = D.MD001 --拆產品組合
            LEFT JOIN COPTJ J ON H.TH074 = J.TJ059 AND H.TH004 = J.TJ004 --作廢紀錄
            INNER JOIN WSCMI I ON G.TG108 = I.MI001 AND LEN(I.MI029) = 10 --會員資訊
            WHERE G.TG004 LIKE 'F909%' --91APP
                AND J.TJ059 IS NULL --若有作廢交易就不納入計算
                AND (P.產品中類代號 = '4201' --判斷鋼瓶產品中類
                OR P.產品大類='氣泡水機')
            """
        return orders_hs_91app_sql_query


    def orders_pos_sql() -> str:
        orders_pos_sql_query = """
            SELECT
                'POS' AS 資料來源,
                V.TV001 AS 日期,
                IIF(D.MD003 IS NULL, V.TV010, D.MD003) AS 品號,
                IIF(D.MD003 IS NULL, V.TV014, V.TV014 * D.MD006) AS 數量,
                V.TV016 AS 消費金額,
                I.MI029 AS 手機,
                品名,
                產品大類
            FROM POSTV V --單頭單身
            LEFT JOIN HLH_DW.dbo.dim_products P ON V.TV010 = P.品號 COLLATE Chinese_Taiwan_Stroke_CI_AS --產品資訊
            LEFT JOIN BOMMD D ON P.產品性質 = '虛擬組合' AND V.TV010 = D.MD001 --拆產品組合
            LEFT JOIN POSTA A ON V.TV001 = A.TA042 AND V.TV002 = A.TA044 AND V.TV003 = A.TA045 AND V.TV006 = A.TA043 --作廢紀錄
            INNER JOIN WSCMI I ON V.TV008 = I.MI001 AND LEN(I.MI029) = 10 --會員資訊
            WHERE V.TV009 IN ('1', '8') --POS 抓銷售、禮券銷售
                AND V.TV016 >= 0 --排除退貨
                AND A.TA001 IS NULL --作廢交易就不納入計算
                AND (P.產品中類代號 = '4201' --判斷鋼瓶產品中類
                OR P.產品大類='氣泡水機')
        """
        return orders_pos_sql_query


    def products_df_sql() -> str:
        products_df_sql_query = """
            SELECT * FROM dim_products
        """
        return products_df_sql_query
