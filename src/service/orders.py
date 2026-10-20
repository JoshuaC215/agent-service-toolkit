
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Order
from db.repositories.order_repository import OrderRepository
from db.session import get_db_session
from schema.orders import (
    OrderCreate,
    OrderResponse,
    OrderStatusUpdate,
)


router = APIRouter(
    prefix="/orders",
    tags=["orders"],
)

DbSession = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order_by_id(
    order_id: str,
    session: DbSession,
) -> OrderResponse:
    repository = OrderRepository(session)
    order = await repository.get_by_id(order_id)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="ORDER_NOT_FOUND",
        )

    return OrderResponse.model_validate(order)


@router.post(
    "",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_order(
    payload: OrderCreate,
    session: DbSession,
) -> OrderResponse:
    repository = OrderRepository(session)

    order = Order(**payload.model_dump())

    try:
        await repository.create(order)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()

        if getattr(exc.orig, "sqlstate", None) == "23505":
            raise HTTPException(
                status_code=409,
                detail="ORDER_ALREADY_EXISTS",
            ) from exc

        raise

    await session.refresh(order)
    return OrderResponse.model_validate(order)


@router.patch(
    "/{order_id}",
    response_model=OrderResponse,
)
async def update_order_status(
    order_id: str,
    payload: OrderStatusUpdate,
    session: DbSession,
) -> OrderResponse:
    repository = OrderRepository(session)

    order = await repository.get_by_id(order_id)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="ORDER_NOT_FOUND",
        )

    await repository.update_status(order, payload.status)
    await session.commit()
    await session.refresh(order)

    return OrderResponse.model_validate(order)