from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.access import AccessDenied, order_scope, principal, require_staff
from db.models import Order


class OrderRepository:
    """Repository for accessing order data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, order_id: str) -> Order | None:
        """Get an order by its primary key."""
        return await self.session.scalar(
            select(Order).where(Order.order_id == order_id, order_scope())
        )

    async def create(self, order: Order) -> Order:
        """Add an order to the database session."""
        require_staff()
        self.session.add(order)
        await self.session.flush()
        return order

    async def list_mine(self, *, limit: int = 20, offset: int = 0) -> tuple[list[Order], bool]:
        """Page only the authenticated customer's orders; never accept an owner argument."""
        identity = principal()
        if identity.role != "customer":
            raise AccessDenied("CUSTOMER_REQUIRED")
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Invalid order pagination")
        result = await self.session.scalars(
            select(Order)
            .where(Order.owner_id == identity.user_id)
            .order_by(Order.created_at.desc(), Order.order_id.desc())
            .offset(offset)
            .limit(limit + 1)
        )
        rows = list(result.all())
        return rows[:limit], len(rows) > limit

    async def update_status(
        self,
        order: Order,
        new_status: str,
    ) -> Order:
        """Update an order's status."""
        require_staff()
        order.status = new_status
        await self.session.flush()
        return order
