from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.access import require_staff, ticket_scope
from db.models import RefundApproval, Ticket


class RefundApprovalRepository:
    """Access refund approvals; the caller owns commit and rollback."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, approval: RefundApproval) -> RefundApproval:
        """Add an approval to the current transaction."""
        self.session.add(approval)
        await self.session.flush()
        return approval

    async def get_by_ticket_id(self, ticket_id: str) -> RefundApproval | None:
        """Get the unique approval associated with a ticket."""
        statement = select(RefundApproval).where(
            RefundApproval.ticket_id == ticket_id,
            RefundApproval.ticket_id.in_(select(Ticket.ticket_id).where(ticket_scope())),
        )
        result = await self.session.scalars(statement)
        return result.one_or_none()

    async def list_pending(self) -> list[RefundApproval]:
        """List pending approvals, oldest first, with stable ordering."""
        require_staff()
        statement = (
            select(RefundApproval)
            .where(RefundApproval.status == "pending")
            .order_by(RefundApproval.created_at.asc(), RefundApproval.approval_id.asc())
        )
        result = await self.session.scalars(statement)
        return list(result.all())
