"""All of Google Cloud Platform related Client are stored in here."""
from types import TracebackType
from typing import TypeVar

from google.cloud.bigquery import Client as BigQueryClient
from google.cloud.bigquery import ConnectionProperty, QueryJobConfig, ScalarQueryParameter
from prefect_gcp import GcpCredentials

BigquerySessionType = TypeVar("BigquerySessionType", bound="BigquerySession")


class BigquerySession:
    """ContextManager wrapping a bigquerySession."""

    def __init__(self: BigquerySessionType, bqclient: BigQueryClient) -> None:
        """Construct instance."""
        self._bigquery_client = bqclient
        self._session_id = None

    def __enter__(self: BigquerySessionType) -> str:
        """Initiate a Bigquery session and return the session_id."""
        job = self._bigquery_client.query(
            "SELECT 1",  # a query can't fail
            job_config=QueryJobConfig(create_session=True),
        )
        self._session_id = job.session_info.session_id
        job.result()  # wait job completion
        return self._session_id

    def __exit__(
            self: BigquerySessionType,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
        """Abort the opened session."""
        if self._session_id:
            # abort the session in any case to have a clean state at the end
            # (sometimes in case of script failure, the table is locked in
            # the session)
            job = self._bigquery_client.query(
                "CALL BQ.ABORT_SESSION()",
                job_config=QueryJobConfig(
                    create_session=False,
                    connection_properties=[
                        ConnectionProperty(
                            key="session_id", value=self._session_id,
                        ),
                    ],
                ),
            )
            job.result()


def get_bigquery_client() -> BigQueryClient:
    """Return BigQuery client."""
    return GcpCredentials.load("datawarehouse").get_bigquery_client(location="asia-east1")


def get_bigquery_client_for_jupyter() -> BigQueryClient:
    """Return BigQuery client."""
    return GcpCredentials.load("datawarehouse").get_bigquery_client(location="asia-east1")


def execute_multiple_query(
    bigquery_client: BigQueryClient,
    queries: list[str],
    query_parameters: list[ScalarQueryParameter] | None = None,
) -> None:
    """Execute multiple SQLs."""
    if query_parameters is None:
        query_parameters = []

    with BigquerySession(bigquery_client) as session_id:
        for query in queries:
            job = bigquery_client.query(
                query,
                job_config=QueryJobConfig(
                    query_parameters=query_parameters,
                    create_session=False,
                    connection_properties=[
                        ConnectionProperty(
                            key="session_id", value=session_id,
                        ),
                    ],
                ),
            )
            job.result()
