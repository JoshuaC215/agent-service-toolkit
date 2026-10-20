from uuid import uuid4

from langchain_core.tools import tool
from sqlalchemy.exc import IntegrityError

from db.models import Ticket
from db.repositories.order_repository import OrderRepository
from db.repositories.ticket_repository import TicketRepository
from db.session import AsyncSessionLocal


def _ticket_to_dict(ticket: Ticket) -> dict:
    """Convert a Ticket ORM object into a JSON-friendly dictionary."""
    return {
        "ticket_id": ticket.ticket_id,
        "order_id": ticket.order_id,
        "issue_type": ticket.issue_type,
        "description": ticket.description,
        "status": ticket.status,
        "created_at": (
            ticket.created_at.isoformat()
            if ticket.created_at
            else None
        ),
        "updated_at": (
            ticket.updated_at.isoformat()
            if ticket.updated_at
            else None
        ),
    }


@tool
async def create_ticket(
    order_id: str,
    issue_type: str,
    description: str,
) -> dict:
    """Create a support ticket for an existing order.

    Use this tool when the user explicitly wants to submit an
    after-sales request, such as a refund request or shipping issue.
    The order must exist. This tool records the request only; it
    does not approve or execute a refund.
    """

    order_id = order_id.strip()
    issue_type = issue_type.strip().lower()
    description = description.strip()

    if not order_id or not issue_type or not description:
        return {
            "success": False,
            "error": "INVALID_TICKET_INPUT",
        }

    async with AsyncSessionLocal() as session:
        order_repository = OrderRepository(session)
        order = await order_repository.get_by_id(order_id)

        if order is None:
            return {
                "success": False,
                "error": "ORDER_NOT_FOUND",
            }

        # Generate the ticket ID on the server.
        ticket = Ticket(
            ticket_id=f"TKT{uuid4().hex[:12].upper()}",
            order_id=order_id,
            issue_type=issue_type,
            description=description,
            status="open",
        )

        repository = TicketRepository(session)

        try:
            await repository.create(ticket)
            await session.commit()
            await session.refresh(ticket)

        except IntegrityError as exc:
            await session.rollback()

            sqlstate = (
                getattr(exc.orig, "sqlstate", None)
                or getattr(exc.orig, "pgcode", None)
            )

            if sqlstate == "23503":
                return {
                    "success": False,
                    "error": "ORDER_NOT_FOUND",
                }

            if sqlstate == "23505":
                return {
                    "success": False,
                    "error": "TICKET_ID_CONFLICT",
                }

            return {
                "success": False,
                "error": "TICKET_CREATE_FAILED",
            }

        return {
            "success": True,
            "ticket": _ticket_to_dict(ticket),
        }


@tool
async def get_ticket(ticket_id: str) -> dict:
    """Get an existing support ticket by its exact ticket ID."""

    ticket_id = ticket_id.strip()

    async with AsyncSessionLocal() as session:
        repository = TicketRepository(session)
        ticket = await repository.get_by_id(ticket_id)

        if ticket is None:
            return {
                "found": False,
                "error": "TICKET_NOT_FOUND",
            }

        return {
            "found": True,
            "ticket": _ticket_to_dict(ticket),
        }


@tool
async def list_tickets_by_order(order_id: str) -> dict:
    """List all support tickets associated with an order ID."""

    order_id = order_id.strip()

    async with AsyncSessionLocal() as session:
        order_repository = OrderRepository(session)
        order = await order_repository.get_by_id(order_id)

        if order is None:
            return {
                "found": False,
                "error": "ORDER_NOT_FOUND",
            }

        repository = TicketRepository(session)
        tickets = await repository.list_by_order_id(order_id)

        return {
            "found": True,
            "order_id": order_id,
            "count": len(tickets),
            "tickets": [
                _ticket_to_dict(ticket)
                for ticket in tickets
            ],
        }