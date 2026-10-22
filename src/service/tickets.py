from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.access import require_staff
from db.models import Ticket
from db.repositories.order_repository import OrderRepository
from db.repositories.ticket_repository import TicketRepository
from db.session import get_db_session
from schema.tickets import (
    TicketCreate,
    TicketResponse,
    TicketStatusUpdate,
)

router = APIRouter(
    prefix="/tickets",
    tags=["tickets"],
)

DbSession = Annotated[
    AsyncSession,
    Depends(get_db_session),
]


@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_ticket(
    payload: TicketCreate,
    session: DbSession,
) -> TicketResponse:
    """Create a support ticket for an existing order."""

    # Verify that the referenced order exists.
    order_repository = OrderRepository(session)
    order = await order_repository.get_by_id(payload.order_id)

    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ORDER_NOT_FOUND",
        )

    repository = TicketRepository(session)
    ticket = Ticket(**payload.model_dump())

    try:
        await repository.create(ticket)
        await session.commit()

    except IntegrityError as exc:
        await session.rollback()

        sqlstate = getattr(exc.orig, "sqlstate", None) or getattr(exc.orig, "pgcode", None)

        if sqlstate == "23505":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="TICKET_ALREADY_EXISTS",
            ) from exc

        if sqlstate == "23503":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="ORDER_NOT_FOUND",
            ) from exc

        raise

    except Exception:
        await session.rollback()
        raise

    await session.refresh(ticket)

    return TicketResponse.model_validate(ticket)


@router.get(
    "/by-order/{order_id}",
    response_model=list[TicketResponse],
)
async def list_tickets_by_order(
    order_id: str,
    session: DbSession,
) -> list[TicketResponse]:
    """List all tickets belonging to an order."""

    order_repository = OrderRepository(session)
    order = await order_repository.get_by_id(order_id)

    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ORDER_NOT_FOUND",
        )

    repository = TicketRepository(session)
    tickets = await repository.list_by_order_id(order_id)

    return [TicketResponse.model_validate(ticket) for ticket in tickets]


@router.get(
    "/{ticket_id}",
    response_model=TicketResponse,
)
async def get_ticket_by_id(
    ticket_id: str,
    session: DbSession,
) -> TicketResponse:
    """Get a ticket by its ID."""

    repository = TicketRepository(session)
    ticket = await repository.get_by_id(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="TICKET_NOT_FOUND",
        )

    return TicketResponse.model_validate(ticket)


@router.patch(
    "/{ticket_id}",
    response_model=TicketResponse,
)
async def update_ticket_status(
    ticket_id: str,
    payload: TicketStatusUpdate,
    session: DbSession,
) -> TicketResponse:
    """Update a ticket's status."""

    require_staff()
    repository = TicketRepository(session)
    ticket = await repository.get_by_id(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="TICKET_NOT_FOUND",
        )

    try:
        await repository.update_status(
            ticket,
            payload.status,
        )
        await session.commit()

    except Exception:
        await session.rollback()
        raise

    await session.refresh(ticket)

    return TicketResponse.model_validate(ticket)
