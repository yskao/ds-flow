"""Credential for Google Sheets."""
import pygsheets
from prefect_gcp.secret_manager import GcpSecret
from pygsheets.client import Client

GOOGLE_SHEETS_SECRET_JSON_STRING = GcpSecret.load("googlesheets").read_secret()


def get_google_sheet_client() -> Client:
    """Get Google Sheets client."""
    return pygsheets.authorize(service_account_json=GOOGLE_SHEETS_SECRET_JSON_STRING)
