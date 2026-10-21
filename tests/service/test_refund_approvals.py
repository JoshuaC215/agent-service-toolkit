from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from core.settings import settings
from db.refund_approval_service import ApprovalError
from service.refund_approvals import get_db_session, router


@pytest.mark.asyncio
async def test_approval_auth_validation_and_conflicts(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    session = AsyncMock()

    async def override_session():
        yield session

    app.dependency_overrides[get_db_session] = override_session
    monkeypatch.setattr(settings, "APPROVAL_REVIEWERS", {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/refund-approvals/pending")).status_code == 503
        monkeypatch.setattr(settings, "APPROVAL_REVIEWERS", {"staff-1": SecretStr("a" * 40)})
        assert (await client.get("/refund-approvals/pending")).status_code == 401
        headers = {"X-Approval-Key": "a" * 40}
        response = await client.post(
            "/refund-approvals/by-ticket/T1/review",
            headers=headers,
            json={"decision": "approved", "reason": "ok", "reviewer_id": "forged"},
        )
        assert response.status_code == 422
        response = await client.post(
            "/refund-approvals/by-ticket/T1/review",
            headers=headers,
            json={"decision": "approved", "reason": "   "},
        )
        assert response.status_code == 422
        review = AsyncMock(side_effect=ApprovalError("APPROVAL_ALREADY_REVIEWED"))
        monkeypatch.setattr("service.refund_approvals.RefundApprovalService.review", review)
        response = await client.post(
            "/refund-approvals/by-ticket/T1/review",
            headers=headers,
            json={"decision": "approved", "reason": " evidence checked "},
        )
        assert response.status_code == 409
        review.assert_awaited_once_with(
            "T1", decision="approved", reviewer_id="staff-1", reason="evidence checked"
        )


@pytest.mark.asyncio
async def test_missing_approval_and_ticket(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    async def override_session():
        yield AsyncMock()

    app.dependency_overrides[get_db_session] = override_session
    monkeypatch.setattr(settings, "APPROVAL_REVIEWERS", {"staff": SecretStr("b" * 40)})
    monkeypatch.setattr(
        "service.refund_approvals.RefundApprovalRepository.get_by_ticket_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "service.refund_approvals.RefundApprovalService.submit",
        AsyncMock(side_effect=ApprovalError("TICKET_NOT_FOUND")),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Approval-Key": "b" * 40},
    ) as client:
        assert (await client.get("/refund-approvals/by-ticket/missing")).status_code == 404
        assert (await client.post("/refund-approvals/by-ticket/missing")).status_code == 404
