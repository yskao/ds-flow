
import logging

from psi.utils.sql import get_gsheet_url_dict
from pygsheets.spreadsheet import Spreadsheet
from utilities import common_tools as ct

logging.basicConfig(level=logging.INFO)


class AuthGoogleSheets:
    def __init__(self, department: int, year: int) -> None:
        self.department = department
        self.url_dict = get_gsheet_url_dict(self.department, year)
        self.google_client = ct.connect_to_gc()

    def update_google_sheets_auth(
        self, email: str, role: str, sheets: list[Spreadsheet],
    ) -> None:
        for sheet in sheets:
            sheet.share(email, role=role, type="user")
            logging.info("%s, %s, %s", sheet.title, email, role)

    def auth_sheets_by_urls(self, email: str, role: str, urls: list[str]) -> None:
        sheets = [self.google_client.open_by_url(url) for url in urls]
        self.update_google_sheets_auth(email, role, sheets)

    def auth_all_sheets(self, email: str, role: str) -> None:
        urls = [url for urls in self.url_dict.values() for url in urls]
        self.auth_sheets_by_urls(email, role, urls)
