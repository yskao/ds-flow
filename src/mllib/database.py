"""Connection for databases."""
import json
from urllib.parse import quote_plus

from prefect_gcp.secret_manager import GcpSecret
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Connection

MSSQL_SCHEMA_DATABASE_MAPPING = json.loads(GcpSecret.load("mssql").read_secret())


def connect_to_mssql(
    schema: str, connect_args: dict | None = None,
) -> Connection:
    """Connect to MSSQL."""
    db = MSSQL_SCHEMA_DATABASE_MAPPING.get(schema, {})
    username = quote_plus(db["username"])
    password = quote_plus(db["password"])
    server = db["server"]
    db_name = db["db_name"]

    default_connect_args = {"autocommit": True, "TrustServerCertificate": "yes"}
    connect_args = connect_args | default_connect_args if connect_args else default_connect_args

    return create_engine(
        f"mssql+pyodbc://{username}:{password}@{server}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server",
        connect_args=connect_args,
        fast_executemany=True,
    ).connect()
