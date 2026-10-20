
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Order


class OrderRepository:
    """Repository for accessing order data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, order_id: str) -> Order | None:
        """Get an order by its primary key."""
        return await self.session.get(Order, order_id)
    
    async def create(self, order: Order) -> Order:
        """Add an order to the database session."""
        self.session.add(order)
        await self.session.flush()
        return order

    async def update_status(
        self,
        order: Order,
        new_status: str,
    ) -> Order:
        """Update an order's status."""
        order.status = new_status
        await self.session.flush()
        return order