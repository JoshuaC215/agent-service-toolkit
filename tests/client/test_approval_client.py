from contextlib import nullcontext
from unittest.mock import patch

import httpx
import pytest

from client.approval_client import ApprovalClient, ApprovalClientError


def test_review_request_keeps_identity_on_server():
    def handle(request):
        assert request.headers["X-Approval-Key"] == "review-key"
        assert request.headers["Authorization"] == "Bearer api-key"
        assert request.url.path == "/refund-approvals/by-ticket/T1/review"
        assert request.content == b'{"decision":"approved","reason":"checked"}'
        return httpx.Response(200, json={"status": "approved"})

    with nullcontext(httpx.Client(transport=httpx.MockTransport(handle))) as http:
        with patch("client.approval_client.httpx.Client", return_value=http):
            result = ApprovalClient("http://test", "review-key", "api-key").review(
                "T1", "approved", "checked"
            )
    assert result["status"] == "approved"


@pytest.mark.parametrize(
    "status,detail,expected",
    [
        (401, "INVALID_APPROVAL_CREDENTIALS", "审批密钥无效"),
        (409, "APPROVAL_ALREADY_REVIEWED", "已被审批"),
        (503, "APPROVAL_AUTH_NOT_CONFIGURED", "尚未配置"),
        (422, [{"msg": "invalid"}], "HTTP 422"),
    ],
)
def test_errors_are_actionable_without_exposing_credentials(status, detail, expected):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"detail": detail}))
    with nullcontext(httpx.Client(transport=transport)) as http:
        with patch("client.approval_client.httpx.Client", return_value=http):
            with pytest.raises(ApprovalClientError, match=expected) as error:
                ApprovalClient("http://test", "private-key").pending()
    assert "private-key" not in str(error.value)


def test_timeout_does_not_retry_write():
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("timeout", request=request)

    with nullcontext(httpx.Client(transport=httpx.MockTransport(handle))) as http:
        with patch("client.approval_client.httpx.Client", return_value=http):
            with pytest.raises(ApprovalClientError, match="不要直接重复提交"):
                ApprovalClient("http://test", "private-key").submit("T1")
    assert len(calls) == 1
