"""Refund review transactions, independent of HTTP and agent tools.

Callers must use a dedicated session and obtain reviewer identity from trusted
authentication. This service does not authenticate users or execute refunds.
"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import RefundApproval, Ticket
from db.repositories.refund_approval_repository import RefundApprovalRepository


class ApprovalError(ValueError):
    """A business rule rejected the operation; the message is a stable error code."""


def _required(value: str, limit: int, code: str) -> str:
    value = value.strip()
    if not value or len(value) > limit:
        raise ApprovalError(code)
    return value


class RefundApprovalService:
    """Own commit/rollback for each operation using a dedicated session."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = RefundApprovalRepository(session)

    async def submit(self, ticket_id: str) -> RefundApproval:
        ticket_id = _required(ticket_id, 32, "INVALID_TICKET_ID")
        try:
            # Serialize submissions for this ticket on PostgreSQL. The unique
            # constraint is an additional database guarantee against duplicates.
            ticket = await self.session.scalar(
                select(Ticket).where(Ticket.ticket_id == ticket_id).with_for_update()
            )
            if ticket is None:
                raise ApprovalError("TICKET_NOT_FOUND")
            if ticket.status not in ("open", "in_progress"):
                raise ApprovalError("TICKET_NOT_ACTIVE")
            if await self.repository.get_by_ticket_id(ticket_id) is not None:
                raise ApprovalError("APPROVAL_ALREADY_EXISTS")
            approval = RefundApproval(
                approval_id=f"APR{uuid4().hex[:28].upper()}",
                ticket_id=ticket_id,
                status="pending",
            )
            await self.repository.create(approval)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        await self.session.refresh(approval)
        return approval

    async def review(
        self, ticket_id: str, *, decision: str, reviewer_id: str, reason: str
    ) -> RefundApproval:
        ticket_id = _required(ticket_id, 32, "INVALID_TICKET_ID")
        reviewer_id = _required(reviewer_id, 64, "INVALID_REVIEWER")
        reason = _required(reason, 4000, "INVALID_REVIEW_REASON")
        if decision not in ("approved", "rejected"):
            raise ApprovalError("INVALID_DECISION")
        try:
            ticket = await self.session.scalar(
                select(Ticket).where(Ticket.ticket_id == ticket_id).with_for_update()
            )
            if ticket is None:
                raise ApprovalError("TICKET_NOT_FOUND")
            if ticket.status not in ("open", "in_progress"):
                raise ApprovalError("TICKET_NOT_ACTIVE")
            # The pending predicate is checked by the database during the write,
            # rather than relying on an earlier Python status check.
            result = await self.session.execute(
                update(RefundApproval)
                .where(
                    RefundApproval.ticket_id == ticket_id,
                    RefundApproval.status == "pending",
                )
                .values(
                    status=decision,
                    reviewer_id=reviewer_id,
                    reason=reason,
                    reviewed_at=datetime.now(UTC),
                )
                .returning(RefundApproval)
            )
            approval = result.scalar_one_or_none()
            if approval is None:
                existing = await self.repository.get_by_ticket_id(ticket_id)
                raise ApprovalError(
                    "APPROVAL_NOT_FOUND" if existing is None else "APPROVAL_ALREADY_REVIEWED"
                )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        await self.session.refresh(approval)
        return approval
