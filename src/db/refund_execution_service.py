"""Atomic database-only simulation, never a real payment workflow."""

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.access import require_staff
from db.models import Order, RefundApproval, RefundExecution, Ticket
from db.refund_approval_service import ApprovalError, _required


class RefundExecutionService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, ticket_id: str) -> RefundExecution | None:
        require_staff()
        ticket_id = _required(ticket_id, 32, "INVALID_TICKET_ID")
        return await self.session.scalar(
            select(RefundExecution)
            .join(RefundApproval, RefundExecution.approval_id == RefundApproval.approval_id)
            .where(RefundApproval.ticket_id == ticket_id)
        )

    async def execute(self, ticket_id: str, executor_id: str) -> tuple[RefundExecution, bool]:
        require_staff()
        ticket_id = _required(ticket_id, 32, "INVALID_TICKET_ID")
        executor_id = _required(executor_id, 64, "INVALID_REVIEWER")
        try:
            ticket = await self.session.scalar(
                select(Ticket).where(Ticket.ticket_id == ticket_id).with_for_update()
            )
            if ticket is None:
                raise ApprovalError("TICKET_NOT_FOUND")
            approval = await self.session.scalar(
                select(RefundApproval)
                .where(RefundApproval.ticket_id == ticket_id)
                .with_for_update()
            )
            if approval is None:
                raise ApprovalError("APPROVAL_NOT_FOUND")
            if approval.status != "approved":
                raise ApprovalError("APPROVAL_NOT_APPROVED")
            existing = await self.get(ticket_id)
            if existing is not None:
                await self.session.commit()
                await self.session.refresh(existing)
                return existing, False
            if ticket.status not in ("open", "in_progress"):
                raise ApprovalError("TICKET_NOT_ACTIVE")
            # Lock the order too: different tickets must not simulate two full refunds.
            order = await self.session.scalar(
                select(Order).where(Order.order_id == ticket.order_id).with_for_update()
            )
            if order is None:
                raise ApprovalError("ORDER_NOT_FOUND")
            if order.status != "paid" or order.amount <= 0:
                raise ApprovalError("ORDER_NOT_REFUNDABLE")
            dialect = self.session.get_bind().dialect.name
            if dialect not in ("postgresql", "sqlite"):
                raise RuntimeError("Unsupported execution database")
            insert = pg_insert if dialect == "postgresql" else sqlite_insert
            execution_id = f"SIM{uuid4().hex[:28].upper()}"
            # Unique constraints arbitrate concurrent requests even without row
            # locks (SQLite). No external side effect occurs outside this transaction.
            result = await self.session.execute(
                insert(RefundExecution)
                .values(
                    execution_id=execution_id,
                    approval_id=approval.approval_id,
                    order_id=order.order_id,
                    amount=order.amount,
                    currency=order.currency,
                    status="simulated_succeeded",
                    executor_id=executor_id,
                    result_message="Simulation succeeded. No funds were transferred.",
                )
                .on_conflict_do_nothing()
                .returning(RefundExecution.execution_id)
            )
            created = result.scalar_one_or_none() is not None
            execution = await self.session.scalar(
                select(RefundExecution).where(RefundExecution.order_id == order.order_id)
            )
            if execution is None:
                raise RuntimeError("Execution record missing")
            if execution.approval_id != approval.approval_id:
                raise ApprovalError("ORDER_ALREADY_SIMULATED")
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        await self.session.refresh(execution)
        return execution, created
