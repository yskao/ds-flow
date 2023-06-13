import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

tz = "Asia/Taipei"
logging.basicConfig(level=logging.INFO)


def get_year_start_and_end(year: int | None = None) -> tuple[date, date]:
    year = year or datetime.now(ZoneInfo(tz)).year
    year_start = datetime.now(ZoneInfo(tz)).replace(year=year, month=1, day=1).date()
    year_end = year_start.replace(month=12)

    return year_start, year_end


def gen_month_str_list(start_date: date, end_date: date) -> list[str]:
    all_months = pd.date_range(start=start_date, end=end_date, freq="MS").date
    return [month.strftime("%Y%m") for month in all_months]


def get_lost_date_in_sorted_date_list(date_list: list[date]) -> list[date]:
    if len(date_list) == 0:
        return []

    date_list.sort()
    date_set = {
        date_list[0] + timedelta(x) for x in range((date_list[-1] - date_list[0]).days)
    }
    return sorted(date_set - set(date_list))


def generate_zodiacal_sign(  # noqa
    input_date: str, str_format: str = "%Y-%m-%d",
) -> str:
    try:
        date_obj = (
            datetime.strptime(input_date, str_format)
            .replace(tzinfo=ZoneInfo(tz))
        )
        month = date_obj.month
        day = date_obj.day
    except ValueError:
        return ""

    if month == 12:
        zodiacal_sign = "射手座" if (day < 22) else "摩羯座"
    elif month == 1:
        zodiacal_sign = "摩羯座" if (day < 20) else "水瓶座"
    elif month == 2:
        zodiacal_sign = "水瓶座" if (day < 19) else "雙魚座"
    elif month == 3:
        zodiacal_sign = "雙魚座" if (day < 21) else "牡羊座"
    elif month == 4:
        zodiacal_sign = "牡羊座" if (day < 20) else "金牛座"
    elif month == 5:
        zodiacal_sign = "金牛座" if (day < 21) else "雙子座"
    elif month == 6:
        zodiacal_sign = "雙子座" if (day < 21) else "巨蟹座"
    elif month == 7:
        zodiacal_sign = "巨蟹座" if (day < 23) else "獅子座"
    elif month == 8:
        zodiacal_sign = "獅子座" if (day < 23) else "處女座"
    elif month == 9:
        zodiacal_sign = "處女座" if (day < 23) else "天秤座"
    elif month == 10:
        zodiacal_sign = "天秤座" if (day < 23) else "天蝎座"
    elif month == 11:
        zodiacal_sign = "天蝎座" if (day < 22) else "射手座"
    else:
        zodiacal_sign = ""
    return zodiacal_sign


def republic_era_date_str_to_ad_date_str(date_str: str) -> str:
    date_split_str = ""
    for split_str in "/.-":
        if split_str in date_str:
            date_split_str = split_str
            break

    if date_split_str:
        date_str_list = date_str.split(date_split_str)
        year = str(int(date_str_list[0]) + 1911)
        month = date_str_list[1]
        day = date_str_list[2]
    else:
        year = str(int(date_str[:-4]) + 1911)
        month = date_str[-4:-2]
        day = date_str[-2:]

    return date_split_str.join([year, month, day])


def is_valid_date(date_str: str, expected_date_format: str = "%Y-%m-%d"):
    try:
        datetime.strptime(date_str, expected_date_format).replace(tzinfo=ZoneInfo(tz))
    except ValueError:
        logging.info("Incorrect date format, should be %s", expected_date_format)
        return False

    return True
