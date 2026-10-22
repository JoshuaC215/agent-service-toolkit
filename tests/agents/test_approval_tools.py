from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from agents import approval_tools
from db.refund_approval_service import ApprovalError


@pytest.fixture
def approval_dependencies(monkeypatch):
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=MagicMock())
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(approval_tools, "AsyncSessionLocal", MagicMock(return_value=context))
    submit = AsyncMock()
    lookup = AsyncMock()
    monkeypatch.setattr(approval_tools.RefundApprovalService, "submit", submit)
    monkeypatch.setattr(approval_tools.RefundApprovalRepository, "get_by_ticket_id", lookup)
    record = SimpleNamespace(
        approval_id="APR1",
        ticket_id="T1",
        status="pending",
        reviewer_id=None,
        reason=None,
        created_at=datetime.now(UTC),
        reviewed_at=None,
    )
    submit.return_value = record
    lookup.return_value = record
    return submit, lookup


@pytest.mark.asyncio
async def test_submit_and_lookup(approval_dependencies):
    submit, lookup = approval_dependencies
    result = await approval_tools.submit_refund_approval.coroutine(ticket_id=" T1 ")
    assert result["success"] is True
    assert result["approval"]["status"] == "pending"
    assert result["refund_executed"] is False
    submit.assert_awaited_once_with("T1")
    result = await approval_tools.get_refund_approval.coroutine(ticket_id="T1")
    assert result["found"] is True
    assert result["refund_execution_status"] == "not_checked"
    lookup.assert_awaited_once_with("T1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["APPROVAL_ALREADY_EXISTS", "TICKET_NOT_FOUND", "TICKET_NOT_ACTIVE"]
)
async def test_business_failure_is_not_success(approval_dependencies, code):
    submit, _ = approval_dependencies
    submit.side_effect = ApprovalError(code)
    result = await approval_tools.submit_refund_approval.coroutine(ticket_id="T1")
    assert result == {"success": False, "error": code}
    assert submit.await_count == 1


@pytest.mark.asyncio
async def test_missing_approval(approval_dependencies):
    _, lookup = approval_dependencies
    lookup.return_value = None
    result = await approval_tools.get_refund_approval.coroutine(ticket_id="T1")
    assert result == {"found": False, "error": "APPROVAL_NOT_FOUND"}


@pytest.mark.asyncio
async def test_db_failure_is_sanitized_and_not_retried(approval_dependencies):
    submit, lookup = approval_dependencies
    submit.side_effect = SQLAlchemyError("private database details")
    lookup.side_effect = SQLAlchemyError("private database details")
    result = await approval_tools.submit_refund_approval.coroutine(ticket_id="T1")
    assert result["error"] == "APPROVAL_SUBMISSION_UNCONFIRMED"
    assert "private" not in str(result)
    assert submit.await_count == 1
    result = await approval_tools.get_refund_approval.coroutine(ticket_id="T1")
    assert result["error"] == "APPROVAL_LOOKUP_UNAVAILABLE"


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket_id", ["  ", "x" * 33])
async def test_invalid_id_never_reaches_service(approval_dependencies, ticket_id):
    submit, lookup = approval_dependencies
    result = await approval_tools.submit_refund_approval.coroutine(ticket_id=ticket_id)
    assert result["error"] == "INVALID_TICKET_ID"
    result = await approval_tools.get_refund_approval.coroutine(ticket_id=ticket_id)
    assert result["error"] == "INVALID_TICKET_ID"
    submit.assert_not_awaited()
    lookup.assert_not_awaited()
