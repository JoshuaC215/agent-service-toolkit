import asyncio
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.models import Base, Order, RefundApproval, Ticket
from db.refund_approval_service import ApprovalError, RefundApprovalService


@pytest_asyncio.fixture
async def sessions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'approvals.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Order(order_id="O1", status="paid", amount=Decimal("10"), currency="CNY"))
        await session.flush()
        session.add(
            Ticket(
                ticket_id="T1",
                order_id="O1",
                issue_type="refund",
                description="test",
                status="open",
            )
        )
        await session.commit()
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approved", "rejected"])
async def test_submit_review_and_prevent_second_review(sessions, decision):
    async with sessions() as session:
        service = RefundApprovalService(session)
        approval = await service.submit("T1")
        assert approval.status == "pending"
        assert approval.reviewer_id is None
        with pytest.raises(ApprovalError, match="APPROVAL_ALREADY_EXISTS"):
            await service.submit("T1")
        reviewed = await service.review(
            "T1", decision=decision, reviewer_id=" staff-1 ", reason=" checked evidence "
        )
        assert reviewed.status == decision
        assert reviewed.reviewer_id == "staff-1"
        assert reviewed.reason == "checked evidence"
        assert reviewed.reviewed_at is not None
        with pytest.raises(ApprovalError, match="APPROVAL_ALREADY_REVIEWED"):
            await service.review("T1", decision="rejected", reviewer_id="staff-2", reason="again")
    async with sessions() as session:
        saved = await session.scalar(select(RefundApproval))
        assert saved.status == decision
        assert saved.reviewer_id == "staff-1"
        assert (await session.get(Ticket, "T1")).status == "open"
        assert (await session.get(Order, "O1")).status == "paid"


@pytest.mark.asyncio
async def test_missing_and_inactive_ticket(sessions):
    async with sessions() as session:
        service = RefundApprovalService(session)
        with pytest.raises(ApprovalError, match="TICKET_NOT_FOUND"):
            await service.submit("missing")
        with pytest.raises(ApprovalError, match="APPROVAL_NOT_FOUND"):
            await service.review("T1", decision="approved", reviewer_id="staff", reason="ok")
        ticket = await session.get(Ticket, "T1")
        ticket.status = "closed"
        await session.commit()
        with pytest.raises(ApprovalError, match="TICKET_NOT_ACTIVE"):
            await service.submit("T1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,code",
    [
        ({"decision": "pending"}, "INVALID_DECISION"),
        ({"reviewer_id": "  "}, "INVALID_REVIEWER"),
        ({"reason": "  "}, "INVALID_REVIEW_REASON"),
    ],
)
async def test_invalid_review_leaves_pending(sessions, change, code):
    async with sessions() as session:
        service = RefundApprovalService(session)
        await service.submit("T1")
        payload = dict(decision="approved", reviewer_id="staff", reason="ok")
        payload.update(change)
        with pytest.raises(ApprovalError, match=code):
            await service.review("T1", **payload)
    async with sessions() as session:
        assert (await session.scalar(select(RefundApproval))).status == "pending"


@pytest.mark.asyncio
async def test_competing_reviewers_only_one_succeeds(sessions):
    async with sessions() as session:
        await RefundApprovalService(session).submit("T1")

    async def review(decision, reviewer):
        async with sessions() as session:
            return await RefundApprovalService(session).review(
                "T1", decision=decision, reviewer_id=reviewer, reason="reviewed"
            )

    results = await asyncio.gather(
        review("approved", "staff-a"), review("rejected", "staff-b"), return_exceptions=True
    )
    assert sum(isinstance(item, RefundApproval) for item in results) == 1
    errors = [item for item in results if isinstance(item, Exception)]
    assert len(errors) == 1
    assert isinstance(errors[0], ApprovalError)
    assert str(errors[0]) == "APPROVAL_ALREADY_REVIEWED"
