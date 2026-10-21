from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
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


@pytest.mark.asyncio
async def test_simulation_auth_identity_confirmation_and_replay(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    async def override_session():
        yield AsyncMock()

    app.dependency_overrides[get_db_session] = override_session
    monkeypatch.setattr(settings, "APPROVAL_REVIEWERS", {"staff": SecretStr("b" * 40)})
    record = SimpleNamespace(
        execution_id="SIM1",
        approval_id="A1",
        order_id="O1",
        amount=Decimal("299.00"),
        currency="CNY",
        status="simulated_succeeded",
        executor_id="staff",
        result_message="No funds transferred",
        executed_at=datetime.now(UTC),
    )
    execute = AsyncMock(return_value=(record, False))
    monkeypatch.setattr("service.refund_approvals.RefundExecutionService.execute", execute)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        path = "/refund-approvals/by-ticket/T1/simulate-refund"
        assert (await client.post(path, json={"confirm_simulation": True})).status_code == 401
        execute.assert_not_awaited()
        headers = {"X-Approval-Key": "b" * 40}
        assert (await client.post(path, headers=headers, json={})).status_code == 422
        assert (
            await client.post(
                path, headers=headers, json={"confirm_simulation": True, "executor_id": "forged"}
            )
        ).status_code == 422
        response = await client.post(path, headers=headers, json={"confirm_simulation": True})
        assert response.status_code == 200
        assert response.json()["created"] is False
        assert response.json()["funds_transferred"] is False
        execute.assert_awaited_once_with("T1", "staff")
        execute.side_effect = ApprovalError("APPROVAL_NOT_APPROVED")
        assert (
            await client.post(path, headers=headers, json={"confirm_simulation": True})
        ).status_code == 409
