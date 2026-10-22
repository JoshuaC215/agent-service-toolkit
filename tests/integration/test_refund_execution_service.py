import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.access import Principal, current_principal

from db.models import Base, Order, RefundApproval, RefundExecution, Ticket
from db.refund_approval_service import ApprovalError
from db.refund_execution_service import RefundExecutionService


@pytest_asyncio.fixture
async def sessions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'execution.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Order(order_id="O1", status="paid", amount=Decimal("299.00"), currency="CNY"))
        await session.flush()
        for number in (1, 2):
            session.add(
                Ticket(
                    ticket_id=f"T{number}",
                    order_id="O1",
                    issue_type="refund",
                    description="test",
                    status="open",
                )
            )
        await session.flush()
        for number in (1, 2):
            session.add(
                RefundApproval(
                    approval_id=f"A{number}",
                    ticket_id=f"T{number}",
                    status="approved",
                    reviewer_id="reviewer",
                    reason="checked",
                    reviewed_at=datetime.now(UTC),
                )
            )
        await session.commit()
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_execution_persists_and_repeat_returns_same_record(sessions):
    async with sessions() as session:
        service = RefundExecutionService(session)
        record, created = await service.execute("T1", "staff-1")
        assert created
        execution_id = record.execution_id
        record, created = await service.execute("T1", "staff-2")
        assert not created
        assert record.execution_id == execution_id
        assert record.executor_id == "staff-1"
    async with sessions() as session:
        record = await RefundExecutionService(session).get("T1")
        assert record.amount == Decimal("299.00")
        assert record.currency == "CNY"
        assert record.status == "simulated_succeeded"
        assert (await session.get(Order, "O1")).status == "paid"
        assert (await session.get(Ticket, "T1")).status == "open"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "rejected"])
async def test_only_approved_can_execute(sessions, status):
    async with sessions() as session:
        approval = await session.get(RefundApproval, "A1")
        approval.status = status
        if status == "pending":
            approval.reviewer_id = approval.reason = approval.reviewed_at = None
        await session.commit()
        with pytest.raises(ApprovalError, match="APPROVAL_NOT_APPROVED"):
            await RefundExecutionService(session).execute("T1", "staff")
        assert await session.scalar(select(func.count()).select_from(RefundExecution)) == 0


@pytest.mark.asyncio
async def test_other_ticket_cannot_refund_same_order(sessions):
    async with sessions() as session:
        await RefundExecutionService(session).execute("T1", "staff")
    async with sessions() as session:
        with pytest.raises(ApprovalError, match="ORDER_ALREADY_SIMULATED"):
            await RefundExecutionService(session).execute("T2", "staff")
        assert await session.scalar(select(func.count()).select_from(RefundExecution)) == 1


@pytest.mark.asyncio
async def test_concurrent_requests_produce_one_record(sessions):
    async def execute():
        async with sessions() as session:
            record, created = await RefundExecutionService(session).execute("T1", "staff")
            return record.execution_id, created

    results = await asyncio.gather(execute(), execute())
    assert results[0][0] == results[1][0]
    assert sum(created for _, created in results) == 1


@pytest.mark.asyncio
async def test_commit_failure_rolls_back_simulation(sessions):
    async with sessions() as session:
        with patch.object(session, "commit", side_effect=RuntimeError("test failure")):
            with pytest.raises(RuntimeError, match="test failure"):
                await RefundExecutionService(session).execute("T1", "staff")
    async with sessions() as session:
        assert await RefundExecutionService(session).get("T1") is None


@pytest.mark.asyncio
async def test_missing_ticket(sessions):
    async with sessions() as session:
        with pytest.raises(ApprovalError, match="TICKET_NOT_FOUND"):
            await RefundExecutionService(session).execute("missing", "staff")


@pytest.mark.asyncio
@pytest.mark.parametrize("status,amount", [("cancelled", "299"), ("paid", "0")])
async def test_invalid_order_does_not_create_execution(sessions, status, amount):
    async with sessions() as session:
        order = await session.get(Order, "O1")
        order.status = status
        order.amount = Decimal(amount)
        await session.commit()
        with pytest.raises(ApprovalError, match="ORDER_NOT_REFUNDABLE"):
            await RefundExecutionService(session).execute("T1", "staff")
        assert await RefundExecutionService(session).get("T1") is None


@pytest.fixture(autouse=True)
def trusted_staff_identity():
    token = current_principal.set(Principal('test-staff', 'staff'))
    try:
        yield
    finally:
        current_principal.reset(token)
