
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Ticket


class TicketRepository:
    """Repository for accessing ticket data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, ticket: Ticket) -> Ticket:
        """Add a ticket to the database session."""
        self.session.add(ticket)
        await self.session.flush()
        return ticket

    async def get_by_id(self, ticket_id: str) -> Ticket | None:
        """Get a ticket by its primary key."""
        return await self.session.get(Ticket, ticket_id)

    async def list_by_order_id(
        self,
        order_id: str,
    ) -> list[Ticket]:
        """Get all tickets associated with an order."""
        statement = (
            select(Ticket)
            .where(Ticket.order_id == order_id)
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
        ticket.status = new_status
        await self.session.flush()
        return ticket