from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.access import require_staff, ticket_scope
from db.models import Ticket


class TicketRepository:
    """Repository for accessing ticket data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, ticket: Ticket) -> Ticket:
        """Add a ticket to the database session."""
        from db.repositories.order_repository import OrderRepository

        if await OrderRepository(self.session).get_by_id(ticket.order_id) is None:
            from db.access import AccessDenied

            raise AccessDenied("ORDER_NOT_FOUND")
        self.session.add(ticket)
        await self.session.flush()
        return ticket

    async def get_by_id(self, ticket_id: str) -> Ticket | None:
        """Get a ticket by its primary key."""
        return await self.session.scalar(
            select(Ticket).where(Ticket.ticket_id == ticket_id, ticket_scope())
        )

    async def list_by_order_id(
        self,
        order_id: str,
    ) -> list[Ticket]:
        """Get all tickets associated with an order."""
        statement = (
            select(Ticket)
            .where(Ticket.order_id == order_id)
            .where(ticket_scope())
            .order_by(Ticket.created_at.desc())
        )

        result = await self.session.scalars(statement)
        return list(result.all())

    async def update_status(
        self,
        ticket: Ticket,
        new_status: str,
    ) -> Ticket:
        """Update a ticket's status."""
        require_staff()
        ticket.status = new_status
        await self.session.flush()
        return ticket
