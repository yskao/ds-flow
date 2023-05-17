"""
Power BI REST API用戶端類別,提供刷新資料集的功能.

請確保您已經進行以下設定:
1. 在 Azure 建立 APP Registration, 取得 tenant_id, client_id, client_secret.
2. 在 Azure 的 APP Registration 設定 API permissions.
3. 在 Azure 的 Active Directory 建立 Security Group,
    並將先前建立的 APP Registration 放進 Security Group.
4. 在 Power BI Admin Portal 開啟 允許服務主體使用 Power BI API,
    並將先前建立的 Security Group 加入到特定的安全性群組.
5. 在 Power BI Workspace 加入 Security Group 並將其權限設定成系統管理員.
6. 在 GCP Secrets Manager 中設定了 Power BI 的憑證和設定,並在使用本類別前先獲取相關的值.
"""
import json
import time
from typing import TypeVar

import msal
import requests
from prefect_gcp.secret_manager import GcpSecret

HTTP_ACCEPTED_CODE = 202
PowerBIClientType = TypeVar("PowerBIClientType", bound="PowerBIClient")


class AccessTokenError(Exception):
    """當Power BI REST API的Access Token無效時拋出的例外類別."""


class DatasetRefreshError(Exception):
    """當Power BI REST API 重新整理資料集失敗的例外類別."""


class RequestIdNotFoundError(Exception):
    """當指定的RequestId不存在時拋出的例外類別."""


class PowerBIClient:
    """
    Power BI REST API用戶端類別,提供刷新資料集的功能.

    請確保您已經在GCP Secrets Manager中設定了Power BI的憑證和設定,並在使用本類別前先獲取相關的值。
    """

    def __init__(self: PowerBIClientType) -> None:
        """
        初始化PowerBIClient實例.

        從GCP Secrets Manager讀取Power BI相關的憑證和設定,並設定相關的權限範圍和API認證的參數。
        """
        self.powerbi_secrets = json.loads(GcpSecret.load("powerbi").read_secret())
        self.tenant_id = self.powerbi_secrets["tenant_id"]
        self.client_id = self.powerbi_secrets["client_id"]
        self.client_secret = self.powerbi_secrets["client_secret"]
        self.scopes = ["https://analysis.windows.net/powerbi/api/.default"]
        self.app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
        )
        self.base_endpoint = "https://api.powerbi.com/v1.0/myorg"
        self.headers = self._get_headers()

    def _get_access_token(self: PowerBIClientType) -> str:
        """
        獲取Power BI REST API的Access Token.

        Returns
        -------
            str: Power BI REST API的Access Token.
        """
        token_response = self.app.acquire_token_for_client(scopes=self.scopes)
        if "access_token" not in token_response:
            raise AccessTokenError(token_response["error_description"])
        return token_response["access_token"]

    def _get_headers(self: PowerBIClientType) -> dict:
        return {"Authorization": f"Bearer {self._get_access_token()}"}

    def refresh_dataset(
        self: PowerBIClientType,
        group_id: str,
        dataset_id: str,
    ) -> str:
        """
        刷新指定的資料集.

        Args:
        ----
            group_id (str): 資料集所在的群組ID.
            dataset_id (str): 資料集的ID.
        """
        endpoint = f"{self.base_endpoint}/groups/{group_id}/datasets/{dataset_id}/refreshes"
        response = requests.post(endpoint, headers=self.headers, timeout=600)
        if response.status_code == HTTP_ACCEPTED_CODE:
            print("Successfully started refreshing the dataset!")  # noqa: T201
        else:
            print(response.reason)  # noqa: T201
            print(response.json())  # noqa: T201
            raise DatasetRefreshError(group_id + "_" + dataset_id)
        return response.headers["RequestId"]

    def get_refresh_status(
        self: PowerBIClientType,
        group_id: str,
        dataset_id: str,
        request_id: str,
    ) -> None:
        """
        獲取 refresh dataset 的狀態.

        Args:
        ----
            group_id (str): 資料集所在的群組ID.
            dataset_id (str): 資料集的ID.
            request_id (str): refresh 任務的 ID.
        """
        endpoint = f"{self.base_endpoint}/groups/{group_id}/datasets/{dataset_id}/refreshes"
        while True:
            response = requests.get(endpoint, headers=self.headers, timeout=600)
            refreshs = {refresh["requestId"]: refresh for refresh in response.json()["value"]}
            refresh = refreshs.get(request_id)
            if not refresh:
                raise RequestIdNotFoundError(request_id)
            if refresh["status"] == "Completed":
                print("Successfully refreshed the dataset!")  # noqa: T201
                break
            time.sleep(30)
            print("Refreshing dataset...")  # noqa: T201


# 使用範例
if __name__ == "__main__":
    group_id = "6727dd2e-db8b-401c-a174-e76725e3660c"
    dataset_id = "0d623926-fc49-4b40-964d-d5db4127b480"
    client = PowerBIClient()
    request_id = client.refresh_dataset(group_id, dataset_id)
    client.get_refresh_status(group_id, dataset_id, request_id)
