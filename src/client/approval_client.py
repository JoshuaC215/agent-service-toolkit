"""HTTP access for the human approval screen; credentials are never cached."""

from urllib.parse import quote

import httpx


class ApprovalClientError(Exception):
    pass


class ApprovalClient:
    def __init__(self, base_url: str, approval_key: str, auth_secret: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-Approval-Key": approval_key}
        if auth_secret:
            self.headers["Authorization"] = f"Bearer {auth_secret}"

    def request(self, method: str, path: str, **kwargs):
        try:
            with httpx.Client(timeout=20, trust_env=False) as client:
                response = client.request(
                    method, f"{self.base_url}{path}", headers=self.headers, **kwargs
                )
        except httpx.RequestError as exc:
            raise ApprovalClientError(
                "无法确认请求结果。请查询最新审批状态后再操作，不要直接重复提交。"
            ) from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                detail = ""
            messages = {
                "APPROVAL_AUTH_NOT_CONFIGURED": "后端尚未配置有效的审批密钥。",
                "INVALID_APPROVAL_CREDENTIALS": "审批密钥无效，请重新输入。",
                "APPROVAL_ALREADY_REVIEWED": "该申请已被审批，请刷新查看结果。",
                "APPROVAL_ALREADY_EXISTS": "该工单已有审批记录，请查询现有记录。",
                "APPROVAL_NOT_FOUND": "该工单尚未提交审批。",
                "TICKET_NOT_FOUND": "工单不存在，请核对编号。",
                "TICKET_NOT_ACTIVE": "工单已不处于可处理状态。",
            }
            message = messages.get(detail) if isinstance(detail, str) else None
            if not message and response.status_code == 401:
                message = "身份验证失败，请检查审批密钥及前端 AUTH_SECRET 配置。"
            raise ApprovalClientError(message or f"请求失败（HTTP {response.status_code}）。")
        return response.json()

    @staticmethod
    def ticket_path(ticket_id: str) -> str:
        return f"/refund-approvals/by-ticket/{quote(ticket_id, safe='')}"

    def pending(self):
        return self.request("GET", "/refund-approvals/pending")

    def get(self, ticket_id: str):
        return self.request("GET", self.ticket_path(ticket_id))

    def submit(self, ticket_id: str):
        return self.request("POST", self.ticket_path(ticket_id))

    def review(self, ticket_id: str, decision: str, reason: str):
        return self.request(
            "POST",
            f"{self.ticket_path(ticket_id)}/review",
            json={"decision": decision, "reason": reason},
        )

    def ticket(self, ticket_id: str):
        return self.request("GET", f"/tickets/{quote(ticket_id, safe='')}")

    def order(self, order_id: str):
        return self.request("GET", f"/orders/{quote(order_id, safe='')}")
