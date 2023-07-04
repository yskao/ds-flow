# %%
import logging
from collections import defaultdict
from datetime import date, datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas import DataFrame
from psi.utils.config import P03_WAREHOUSE_DF
from pygsheets.spreadsheet import Spreadsheet
from utilities.database import connect_to_mssql
from utilities.google_sheets import get_google_sheet_client

logging.basicConfig(level=logging.INFO)

# %%
def get_dent_sales() -> pd.DataFrame:
    """取得牙醫通路業務的員工編號,欄位: 業務員代號."""
    with connect_to_mssql("hlh") as conn:
        sql = """
            SELECT
                RTRIM(empid) AS 業務員代號
            FROM Flow.dbo.OralB_RPT_Area
        """
        dent_sales = pd.read_sql(sql, conn)

    logging.info("取得牙醫通路業務的員工編號")

    return dent_sales


# 查詢庫存月報表資料
def get_inventory_data(
    data_ym: datetime, department_code: str, compute_case: str,
) -> DataFrame:
    """
    input範例
    data_ym: datetime
    department_code: 'P03',
    compute_case: 'P' (可以是'P'或'I').
    """
    logging.info("撈取庫存月報表: {data_ym_str}", extra={"data_ym_str": data_ym.strftime("%Y%m")})
    logging.info("計算處室: {department_code}", extra={"department_code": department_code})
    logging.info("計算類別: {compute_case}", extra={"compute_case": compute_case})

    department = department_code[2:3]
    warehouse_p_list = [
        f"{warehouse_code}{department}"
        for warehouse_code in ["A01", "XR1", "XR2", "XW4", "XW5"]
    ]
    if department == "3":
        warehouse_hs_list = ["XR11HS", "XW41HS"]
        warehouse_dict = {
            "P": {
                "A_LA009_filter": "substring(A.LA009,1,4) in %(warehouse_p_list)s",
                "LA009_filter": "substring(LA009,1,4) in %(warehouse_p_list)s",
                "LC003_filter": "substring(LC003,1,4) in %(warehouse_p_list)s",
            },
            "I": {
                "A_LA009_filter": "substring(A.LA009,4,1) = %(department)s or A.LA009 in %(warehouse_hs_list)s",
                "LA009_filter": "substring(LA009,4,1) = %(department)s or  LA009 in %(warehouse_hs_list)s",
                "LC003_filter": "substring(LC003,4,1) = %(department)s or  LC003 in %(warehouse_hs_list)s",
            },
        }
    elif department == "4":
        warehouse_hs_list = ["XR14HS"]
        warehouse_p_list += ["A104", "A054", "A174"]
        warehouse_dict = {
            "P": {
                "A_LA009_filter": "substring(A.LA009,1,4) in %(warehouse_p_list)s or A.LA009 in %(warehouse_hs_list)s",
                "LA009_filter": "substring(LA009,1,4) in %(warehouse_p_list)s or  LA009 in %(warehouse_hs_list)s",
                "LC003_filter": "substring(LC003,1,4) in %(warehouse_p_list)s or  LC003 in %(warehouse_hs_list)s",
            },
            "I": {
                "A_LA009_filter": "substring(A.LA009,4,1) = %(department)s or A.LA009 in %(warehouse_hs_list)s",
                "LA009_filter": "substring(LA009,4,1) = %(department)s or  LA009 in %(warehouse_hs_list)s",
                "LC003_filter": "substring(LC003,4,1) = %(department)s or  LC003 in %(warehouse_hs_list)s",
            },
        }
    else:
        raise

    a_la009_filter = warehouse_dict[compute_case]["A_LA009_filter"]
    la009_filter = warehouse_dict[compute_case]["LA009_filter"]
    lc003_filter = warehouse_dict[compute_case]["LC003_filter"]

    sql = f"""SET NOCOUNT ON;  --不加這個pdc會報錯

    DECLARE  @Date varchar(6) = %(date_str)s --年月YYYYMM
    DECLARE  @DepCheck varchar(2) =  %(department)s --事業處,計算銷貨的時候用到,取那個事業處的單子

    DECLARE @TmpTable TABLE(
    [料號] [nvarchar](40) NULL,
    [品名] [nvarchar](60) NULL,
    [中類別名稱] [nvarchar](12) NULL,
    [期初存貨] [numeric](38, 3) NULL,
    [本期進貨] [numeric](38, 3) NOT NULL,
    [本期加工] [numeric](38, 3) NOT NULL,
    [本期調撥] [numeric](38, 3) NOT NULL,
    [本期拆組] [numeric](38, 3) NOT NULL,
    [本期銷貨] [numeric](38, 3) NOT NULL,
    [本期退貨] [numeric](38, 3) NOT NULL,
    [本期損失] [numeric](38, 3) NOT NULL,
    [盤點盈虧] [numeric](38, 3) NOT NULL,
    [本期報廢] [numeric](38, 3) NOT NULL,
    [本期領用] [numeric](38, 3) NOT NULL,
    [本期贈樣] [numeric](38, 3) NOT NULL,
    [本期調整] [numeric](38, 3) NOT NULL,
    [本期結存] [numeric](38, 3) NOT NULL
    )

    insert @TmpTable
    select  *
    from (
        select 料號,
            D.MB002 as '品名',
            F.MA003'中類別名稱',
            sum(期初結存量)'期初存貨',
            sum(本期進貨)'本期進貨',
            sum(本期加工)'本期加工',
            sum(本期調撥)'本期調撥',
            sum(本期拆組)*(-1)'本期拆組',
            sum(本期銷貨)*(-1)'本期銷貨',
            sum(本期退貨)'本期退貨',
            sum(本期損失)*(-1)'本期損失',
            sum(盤點盈虧)'盤點盈虧',
            sum(本期報廢)*(-1)'本期報廢',
            sum(本期領用)*(-1)'本期領用',
            sum(本期贈樣)*(-1)'本期贈樣',
            sum(本期調整)'本期調整',
            sum(O.期初結存量)+sum(O.本期進貨)+sum(O.本期加工)+sum(O.本期調撥)
                -sum(O.本期拆組)*(-1)-sum(O.本期銷貨)*(-1)+sum(O.本期退貨)-sum(O.本期損失)*(-1)
                +sum(O.盤點盈虧)-sum(O.本期報廢)*(-1)-sum(O.本期領用)*(-1)-sum(O.本期贈樣)*(-1)+sum(O.本期調整) as '本期結存'
        from (
            --計算期初結存量
            select Rtrim(料號)'料號',
                SUM(入庫數量)'期初結存量',
                0 as '本期進貨',
                0 as '本期加工',
                0 as '本期調撥',
                0 as '本期拆組',
                0 as '本期銷貨',
                0 as '本期退貨',
                0 as '本期損失',
                0 as '盤點盈虧',
                0 as '本期報廢',
                0 as '本期領用',
                0 as '本期贈樣',
                0 as '本期調整'
            from (
                select Rtrim(LC001)'料號',
                    SUM(LC004)'入庫數量'
                from INVLC
                where LC002=substring(CONVERT(varchar(100),DATEADD(MONTH,-2,@Date+'01'),112),1,6)
                    and ({lc003_filter}) --轉入倉別
                group by LC001
                union all
                select Rtrim(LA001)'料號',
                    ISNULL(LA011,0)'入庫數量'
                from INVLA
                where LA004 >=substring(CONVERT(varchar(100),DATEADD(MONTH,-2,@Date+'01'),112),1,8)
                    and LA004 < (case when @Date='' then '299901' else substring(CONVERT(varchar(100),DATEADD(MONTH,0,@Date+'01'),112),1,8) end)
                    and LA005 = '1'
                    and ({la009_filter})
                union all
                select Rtrim(LA001)'料號',
                    ISNULL(-LA011,0)'入庫數量'
                from INVLA
                where LA004 >=substring(CONVERT(varchar(100),DATEADD(MONTH,-2,@Date+'01'),112),1,8)
                    and LA004 < (case when @Date='' then '299901' else substring(CONVERT(varchar(100),DATEADD(MONTH,0,@Date+'01'),112),1,8) end)
                    and LA005 = '-1'
                    and ({la009_filter})
                ) as Z group by 料號

            UNION ALL
            -- 計算本期資料
            select Rtrim(A.LA001)'料號',
                '0' as '期初結存量',
                SUM(ISNULL(B.LA011,0) * ISNULL(B.LA005,0))'本期進貨',
                SUM(ISNULL(C.LA011,0) * ISNULL(C.LA005,0))'本期加工',
                SUM(ISNULL(D.LA011,0) * ISNULL(D.LA005,0))'本期調撥',
                SUM(ISNULL(E.LA011,0) * ISNULL(E.LA005,0))'本期拆組',
                SUM(ISNULL(F.LA011,0) * ISNULL(F.LA005,0))'本期銷貨',
                SUM(ISNULL(G.LA011,0) * ISNULL(G.LA005,0))'本期退貨',
                SUM(ISNULL(H.LA011,0) * ISNULL(H.LA005,0))'本期損失',
                SUM(ISNULL(I.LA011,0) * ISNULL(I.LA005,0))'盤點盈虧',
                SUM(ISNULL(J.LA011,0) * ISNULL(J.LA005,0))'本期報廢',
                SUM(ISNULL(K.LA011,0) * ISNULL(K.LA005,0))'本期領用',
                SUM(ISNULL(L.LA011,0) * ISNULL(L.LA005,0))'本期贈樣',
                SUM(ISNULL(M.LA011,0) * ISNULL(M.LA005,0))'本期調整'
            from INVLA A
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('3401','3402','3403','3404','3405','3406','3496','3497','3499','3501','3502','3511','3522','3599')
                    ) AS B ON  A.LA005=B.LA005 and A.LA006=B.LA006 and A.LA007=B.LA007 and A.LA008=B.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('4203','4204','4205','4209')
                    ) AS C ON A.LA005=C.LA005 and A.LA006=C.LA006 and A.LA007=C.LA007 and A.LA008=C.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('1109','1201','1207','1208','1209','120A','120B','120C','120D','23B1')
                    ) AS D ON  A.LA005=D.LA005 and A.LA006=D.LA006 and A.LA007=D.LA007 and A.LA008=D.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('4301','4304')
                    ) AS E ON A.LA005=E.LA005 and A.LA006=E.LA006 and A.LA007=E.LA007 and A.LA008=E.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE (LA006 in ('2301','2302','2303','2304','2308','2309','230A','2399','23B6','23B7','23ZZ')
                            AND @DepCheck = '99')
                            OR (@DepCheck <> '99' AND LA006 IN (SELECT          '2301' AS Expr1
                                                                UNION ALL
                                                                SELECT          '2302' AS Expr1
                                                                UNION ALL
                                                                SELECT          '2303' AS Expr1
                                                                UNION ALL
                                                                SELECT          '2304' AS Expr1
                                                                UNION ALL
                                                                SELECT          '2308' AS Expr1
                                                                UNION ALL
                                                                SELECT          '2309' AS Expr1
                                                                UNION ALL
                                                                SELECT          '2399' AS Expr1
                                                                UNION ALL
                                                                SELECT          '23B7' AS Expr1
                                                                UNION ALL
                                                                SELECT          '23ZZ' AS Expr1
                                                                UNION ALL
                                                                SELECT DISTINCT INVLA.LA006
                                                                FROM              INVLA INNER JOIN
                                                                WSCMA ON INVLA.LA006 = WSCMA.MA001
                                                                )) --INVLA.LA006為單別
                    ) AS F ON  A.LA005=F.LA005 and A.LA006=F.LA006 and A.LA007=F.LA007 and A.LA008=F.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('2401','2402','2403','240A','2499','24ZZ')
                    ) AS G ON A.LA005=G.LA005 and A.LA006=G.LA006 and A.LA007=G.LA007 and A.LA008=G.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('23B2','23B3')
                    ) AS H ON A.LA005=H.LA005 and A.LA006=H.LA006 and A.LA007=H.LA007 and A.LA008=H.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('1110','1111')
                    ) AS I ON A.LA005=I.LA005 and A.LA006=I.LA006 and A.LA007=I.LA007 and A.LA008=I.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('1108','23B9')
                    ) AS J ON A.LA005=J.LA005 and A.LA006=J.LA006 and A.LA007=J.LA007 and A.LA008=J.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('1103','1104')
                    ) AS K ON A.LA005=K.LA005 and A.LA006=K.LA006 and A.LA007=K.LA007 and A.LA008=K.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('1102','23A1','23A2','23A3','23A4','23A5','23AA')
                    ) AS L ON A.LA005=L.LA005 and A.LA006=L.LA006 and A.LA007=L.LA007 and A.LA008=L.LA008
                left JOIN (
                    SELECT *
                    FROM INVLA
                    WHERE LA006 in ('1101','1194','1195','1196','1197','1198','1199','11YY','11ZZ')
                    ) AS M ON A.LA005=M.LA005 and A.LA006=M.LA006 and A.LA007=M.LA007 and A.LA008=M.LA008
            where A.LA004 >=@Date and A.LA004 < ( substring(CONVERT(varchar(100),DATEADD(MONTH,1,@Date+'01'),112),1,8) )
                and ({a_la009_filter})
            --and ( A.LA009 in (@WarehouseCheck))
            group by Rtrim(A.LA001)
        ) O
            inner join INVMB D on O.料號=D.MB001
            left JOIN (SELECT * FROM INVMA WHERE MA001 = '5') AS F ON D.MB111 = F.MA002
        group by 料號,D.MB002,F.MA003
    ) A

    select A.料號 AS 品號,
        A.本期進貨,
        A.本期結存
    from (
        select *
        from @TmpTable
        ) A
    left join INVMB B on A.料號=B.MB001 COLLATE Chinese_Taiwan_Stroke_CI_AS
    where B.MB115 in ('T100','T000')
    order by 中類別名稱 desc ,料號 asc
    """ # noqa

    with connect_to_mssql("hlh") as conn:
        tmp_df = pd.read_sql(
            sql,
            conn,
            params={
                "date_str": data_ym.strftime("%Y%m"),
                "department": department,
                "warehouse_hs_list": warehouse_hs_list,
                "warehouse_p_list": warehouse_p_list,
            },
        )

    return tmp_df


def _merge_brand_info(department: int, sh: Spreadsheet, sales_df: DataFrame) -> DataFrame:

    if department == 3:
        df_brands = sh.worksheet_by_title("品牌資料").get_as_df(numerize=False)
        df_brands = df_brands[["品牌代號", "品牌"]]

    elif department == 4:
        with connect_to_mssql("hlh_dw") as conn:
            sql_brand = """
                SELECT
                    品牌,
                    品牌代號
                FROM dim_brands
            """
            df_brands = pd.read_sql(sql_brand, conn)
        df_brands["品牌"] = df_brands["品牌"].str.replace("SUNBEAM生活", "SUNBEAM")

    else:
        raise

    return sales_df.merge(df_brands, how="left", on="品牌代號")


def _merge_channel_info(sh: Spreadsheet, sales_df: DataFrame) -> DataFrame:

    with connect_to_mssql("hlh_dw") as conn:
        sql_group = """
            SELECT
                客戶代號,
                集團代號
            FROM dim_customers
        """
        group_df = pd.read_sql(sql_group, conn)

    sales_df = sales_df.merge(group_df, how="left", on="客戶代號")
    df_all_employees = sh.worksheet_by_title("業務負責通路資料").get_as_df(numerize=False)

    df_customers = df_all_employees[["業務員代號", "集團代號", "頁籤分類", "通路分類"]].drop_duplicates()
    result_df = sales_df.merge(df_customers, how="left", on=["業務員代號", "集團代號"])

    sales_df_merge_on_customer_code = sales_df.merge(
        df_customers, how="left", left_on=["業務員代號", "客戶代號"], right_on=["業務員代號", "集團代號"],
    )
    result_df["頁籤分類"] = result_df["頁籤分類"].fillna(
        sales_df_merge_on_customer_code["頁籤分類"],
    )
    result_df["頁籤分類"] = result_df["頁籤分類"].fillna("其他")
    result_df["通路分類"] = result_df["通路分類"].fillna(
        sales_df_merge_on_customer_code["通路分類"],
    )
    result_df["通路分類"] = result_df["通路分類"].fillna("其他通路")

    return result_df


def get_sales_data(
    department_code: str,
    start_date: date,
    end_date: date | None = None,
    need_transform: bool = True,
) -> DataFrame:

    end_date = end_date or start_date + relativedelta(months=1)

    sql_sales = """
        SELECT
            日期,
            dim_employees_sales.業務員代號,
            dim_employees_sales.業務員名稱,
            銷貨數量 - 退貨數量 + 銷貨贈品數量 AS 銷貨淨數量,
            銷貨金額 - 退貨金額 AS 銷貨淨金額,
            客戶代號,
            品牌代號,
            品號,
            事業處代號
        FROM dwd_sales_and_returns_all
        LEFT JOIN dim_departments ON dwd_sales_and_returns_all.完整部門代號 = dim_departments.完整部門代號
        LEFT JOIN dim_employees_sales ON dwd_sales_and_returns_all.業務員代號 = dim_employees_sales.業務員代號
        WHERE dim_departments.事業處代號 = %(full_department_code)s
            AND 日期 >= %(start_date_str)s AND 日期 < %(end_date_str)s
            AND 單別 IN ('2301', '2302', '2303', '2308', '2309', '2399', '230A', '2401', '2402', '2403', '2499', '240A', 'POS-PRE', 'POS-NONPRE')
    """
    with connect_to_mssql("hlh_dw") as conn:
        sales_df = pd.read_sql(
            sql_sales,
            conn,
            params={
                "full_department_code": f"{department_code}00",
                "start_date_str": start_date.strftime("%Y%m%d"),
                "end_date_str": end_date.strftime("%Y%m%d"),
            },
        )

    if not need_transform:
        return sales_df

    from psi.baseclass import PSIBase

    department = int(department_code[2])
    base = PSIBase(department)
    sh = base.sh_setting

    # 加入品牌資訊
    sales_df = _merge_brand_info(department, sh, sales_df)

    # 通路資訊
    sales_df = _merge_channel_info(sh, sales_df)

    logging.info(
        "取得 {department}處 在 {start_date} ~ {end_date} 的銷售資料",
        extra={
            "department": department,
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    return sales_df


# 查詢三處調撥給六、七處的資料,處別要調整的話要改庫別篩選設定
def get_warehouse_transfer(
    month_date: date, from_department: int, to_department_list: list[int],
) -> DataFrame:

    logging.info(
        "開始撈取 {from_department}處 在 {month_date_str} 到 {to_department_list} 的調撥資料",
        extra={
            "from_department": from_department,
            "month_date_str": month_date.strftime("%Y%m"),
            "to_department_list": to_department_list,
        },
    )

    next_month = month_date + relativedelta(months=1)
    sql = """
        DECLARE @StartDate int = %(start_dt)s --單據日期起
        DECLARE @EndDate int = %(end_dt)s --單據日期迄
        DECLARE  @TA001 varchar(6) = '1201' --單別
        DECLARE  @TA006 varchar(1) = 'Y' --確認碼,Y/N/V

        SELECT TB004'品號',
            TB005'品名',
            SUM(TB007)'期間轉出總量',
            ISNULL(Z.歸還總量,0)'期間歸還總量',
            (SUM(TB007)-ISNULL(Z.歸還總量,0))'實際轉出總量'
        FROM INVTA A
            LEFT JOIN INVTB BB ON A.TA001=BB.TB001 and  A.TA002=BB.TB002
            LEFT JOIN CMSMC C1 ON BB.TB012=C1.MC001
            LEFT JOIN CMSMC C2 ON BB.TB013=C2.MC001
            LEFT JOIN INVMB ON TB004=MB001
            LEFT JOIN (
                        SELECT TB004'料號',
                            TB005'品名',
                            SUM(TB007)'歸還總量'
                        FROM INVTA A
                            LEFT JOIN INVTB BB ON A.TA001=BB.TB001 and  A.TA002=BB.TB002
                            LEFT JOIN CMSMC C1 ON BB.TB012=C1.MC001
                            LEFT JOIN CMSMC C2 ON BB.TB013=C2.MC001
                            LEFT JOIN INVMB ON TB004=MB001
                        where TA014 >= convert(varchar, @StartDate, 112)
                            and TA014 < convert(varchar, @EndDate, 112)
                            AND TA001 in (@TA001)
                            AND TA006 in (@TA006)
                            AND (substring(TB012, 4, 1) in %(case_list)s) --TB012: 轉出庫
                            AND (substring(TB013, 4, 1) = %(department_str)s or substring(TB013, 4, 3) = '1HS') --TB013: 轉入庫
                        GROUP BY  TB004,TB005
                        )Z ON Z.料號=TB004
        where TA014 >= convert(varchar, @StartDate, 112)
            and TA014 < convert(varchar, @EndDate, 112)
            AND TA001 in (@TA001)
            AND TA006 in (@TA006)
            AND (substring(TB012, 4, 1) = %(department_str)s or substring(TB012, 4, 3) = '1HS') --TB012: 轉出庫
            AND (substring(TB013, 4, 1) in %(case_list)s) --TB013: 轉入庫
        GROUP BY  TB004,TB005,Z.歸還總量
    """
    with connect_to_mssql("hlh") as conn:
        transfer_monthly = pd.read_sql(
            sql,
            conn,
            params={
                "start_dt": int(month_date.strftime("%Y%m%d")),
                "end_dt": int(next_month.strftime("%Y%m%d")),
                "case_list": [str(case) for case in to_department_list],
                "department_str": str(from_department),
            },
        )
        logging.info("取得調撥資料")

    return transfer_monthly


def get_shipping_quantity(month_date: datetime, department: int = 3) -> DataFrame:

    logging.info(
        "開始撈取 {department}處 在 {month_date_str} 的出貨資料",
        extra={
            "department": department,
            "month_date_str": month_date.strftime("%Y%m"),
        },
    )

    warehouse_list = P03_WAREHOUSE_DF["warehouse_code"].to_list()

    next_month = month_date + relativedelta(months=1)
    sql = """
        SELECT
            TB004 '品號',
            TB005 '品名',
            TB013 '庫別代號',
            SUM(TB007) '期間轉出總量',
            ISNULL(Z.歸還總量, 0) '期間歸還總量',
            (SUM(TB007) - ISNULL(Z.歸還總量, 0)) '實際轉出總量'
        FROM
            INVTA A
            LEFT JOIN INVTB BB ON A.TA001 = BB.TB001
            and A.TA002 = BB.TB002
            LEFT JOIN CMSMC C1 ON BB.TB012 = C1.MC001
            LEFT JOIN CMSMC C2 ON BB.TB013 = C2.MC001
            LEFT JOIN INVMB ON TB004 = MB001
            LEFT JOIN (
                SELECT
                    TB004 '料號',
                    TB005 '品名',
                    SUM(TB007) '歸還總量'
                FROM
                    INVTA A
                    LEFT JOIN INVTB BB ON A.TA001 = BB.TB001
                    and A.TA002 = BB.TB002
                    LEFT JOIN CMSMC C1 ON BB.TB012 = C1.MC001
                    LEFT JOIN CMSMC C2 ON BB.TB013 = C2.MC001
                    LEFT JOIN INVMB ON TB004 = MB001
                where
                    TA014 >= %(start_dt)s
                    and TA014 < %(end_dt)s
                    AND TA001 = '120B'
                    AND TA006 = 'Y'
                    AND TB012 in %(warehouse_list)s --TB012: 轉出庫
                    AND (
                        substring(TB013, 4, 1) = %(department_str)s
                        or substring(TB013, 4, 3) = '1HS'
                    ) --TB013: 轉入庫
                GROUP BY
                    TB004,
                    TB005
            ) Z ON Z.料號 = TB004
        WHERE
            TA014 >= %(start_dt)s
            and TA014 < %(end_dt)s
            AND TA001 = '120A'
            AND TA006 = 'Y'
            AND (
                substring(TB012, 4, 1) = %(department_str)s
                or substring(TB012, 4, 3) = '1HS'
            ) --TB012: 轉出庫
            AND TB013 in %(warehouse_list)s --TB013: 轉入庫
        GROUP BY
            TB004,
            TB005,
            TB013,
            Z.歸還總量
    """
    with connect_to_mssql("hlh") as conn:
        transfer_monthly = pd.read_sql(
            sql,
            conn,
            params={
                "start_dt": month_date.strftime("%Y%m%d"),
                "end_dt": next_month.strftime("%Y%m%d"),
                "department_str": str(department),
                "warehouse_list": warehouse_list,
            },
        )
        logging.info("取得調撥資料")

    return transfer_monthly


def get_price_hist(exe_month_date: datetime) -> DataFrame:
    """
    取得歷史成本數據
    欄位: 品號、單位成本.
    """
    with connect_to_mssql("hlh") as conn:
        price_hist = pd.read_sql(
            """
            SELECT LB001 as 品號,
            LB010 as 單位成本
            FROM INVLB
            WHERE LB002 = %(date_str)s
            """,
            conn,
            params={"date_str": exe_month_date.strftime("%Y%m")},
        )
    price_hist["品號"] = price_hist["品號"].str.strip()

    logging.info(
        "取得{exe_month_date}單價資訊",
        extra={
            "exe_month_date": exe_month_date,
        },
    )

    return price_hist


def get_price_realtime() -> DataFrame:
    """
    取得及時單位成本
    欄位:品號、單位成本.
    """
    with connect_to_mssql("hlh") as conn:
        sql = """SELECT RTRIM(MB001) AS 品號,
            MB065 / MB064 AS 單位成本
            FROM INVMB
            WHERE MB064 <> 0
        """
        price_realtime = pd.read_sql(sql, conn)
        sql_new_product = """SELECT RTRIM(MB001) AS 品號,
            (case when MB046 <> 0 then MB046
            when MB050 <> 0 then MB050 end) AS 單位成本
            FROM INVMB
            WHERE MB064 = 0
        """
        price_realtime_new_product = pd.read_sql(sql_new_product, conn)
    price_realtime = pd.concat([price_realtime, price_realtime_new_product], axis=0)
    logging.info("取得及時單價資訊")

    return price_realtime


def get_purchase_price_usd() -> DataFrame:
    """取得最近美金進價."""
    with connect_to_mssql("hlh") as conn:
        sql = """
            SELECT
                RTRIM(MB001) AS 品號,
                MB049 AS 原幣進價
            FROM INVMB
            WHERE MB048 = 'USD'
        """
        price_origin_curr = pd.read_sql(sql, conn)
    logging.info("取得原幣進價")

    return price_origin_curr


def get_department_num(url: str) -> int:
    """根據 url 取得相應部處."""
    gc = get_google_sheet_client()
    sh = gc.open_by_url(url)
    sh_id = sh.id
    with connect_to_mssql("hlh_psi") as conn:
        sql = "SELECT department FROM f_gsheets_info WHERE sh_id = %(sh_id)s"
        result = conn.execute(sql, sh_id=sh_id)
        department = result.fetchone()[0]

    return department


def get_gsheet_url_dict(department: int, year: int) -> dict[str, list[str]]:
    """
    取得DB中 gsheets 資料
    輸入 DB 中的 gsheets 資料,整理所有 url 資料(記錄在self.url_dict).
    """
    sql = """
        SELECT *
        FROM f_gsheets_info
        WHERE department = %(department)s
            AND year = %(year)s
    """

    with connect_to_mssql("hlh_psi") as conn:
        gsheet_df = pd.read_sql(
            sql, conn, params={"department": department, "year": year},
        )

    url_dict = defaultdict(list)
    for _, data in gsheet_df.iterrows():
        url_dict[data["gsheet_type"]].append(data["sh_url"])

    return url_dict


def get_sales_goal_df(department_code: str, year_month_str: str) -> DataFrame:
    with connect_to_mssql("hlh") as conn:
        sql = """
            SELECT yearmonth AS '日期',
                DepId AS '部門代號',
                RealDep,
                SalesId AS '業務員代號',
                Sales AS '業務員名稱',
                target1 AS '業績目標',
                target2 AS '毛利目標',
                CustId AS '客戶代號',
                ChannelId AS '通路代號',
                Brand AS '品牌',
                BrandId AS '品牌代號',
                Category7 AS '目標種類'
            FROM V_ERP_Target
            WHERE yearmonth >= %(year_month_str)s
                AND DepId like %(department_code)s
        """
        return pd.read_sql(
            sql,
            conn,
            params={
                "year_month_str": year_month_str,
                "department_code": f"{department_code}%",
            },
        )


def get_forecast_sales(
    start_date_str: str,
    end_date_str: str,
    full_department_code: str,
) -> DataFrame:
    """取得過去預估資料."""
    with connect_to_mssql("hlh_psi") as conn:
        sql = """
            SELECT *
            FROM f_sales_forecast_versions
            WHERE 日期 >= %(start_date_str)s AND 日期 < %(end_date_str)s
                AND 完整部門代號 = %(full_department_code)s
        """
        return pd.read_sql(
            sql, conn, params={
                "start_date_str": start_date_str,
                "end_date_str": end_date_str,
                "full_department_code": full_department_code,
            },
        )


def upsert_realized_data_by_date(
    table: str,
    realized_df: DataFrame,
    start_date_str: str,
    end_date_str: str,
    full_department_code: str,
) -> None:

    with connect_to_mssql("hlh_psi") as conn:
        delete_sql = f"""
            DELETE FROM {table}
            WHERE 日期 >= %(start_date_str)s AND 日期 < %(end_date_str)s
                AND 完整部門代號 = %(full_department_code)s
        """ # noqa
        conn.execute(
            delete_sql,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            full_department_code=full_department_code,
        )

        logging.info(
            "刪除 {start_date_str} ~ {end_date_str} 的資料",
            extra={
                "start_date_str": start_date_str,
                "end_date_str": end_date_str,
            },
        )

        realized_df.to_sql(
            name=table,
            con=conn,
            if_exists="append",
            index=False,
        )

        logging.info("資料已存入 {table}", extra={"table": table})

# %%
