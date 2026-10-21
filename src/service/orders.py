from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import settings
from db.access import require_staff
from db.models import Order
from db.repositories.order_repository import OrderRepository
from db.session import get_db_session
from schema.orders import (
    MyOrdersQuery,
    MyOrdersResponse,
    OrderCreate,
    OrderOwnerUpdate,
    OrderResponse,
    OrderStatusUpdate,
)

router = APIRouter(
    prefix="/orders",
    tags=["orders"],
)

DbSession = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("/me", response_model=MyOrdersResponse)
async def list_my_orders(filters: Annotated[MyOrdersQuery, Query()], session: DbSession):
    orders, has_more = await OrderRepository(session).list_mine(
        limit=filters.limit, offset=filters.offset
    )
    return MyOrdersResponse(
        orders=[OrderResponse.model_validate(order) for order in orders],
        limit=filters.limit,
        offset=filters.offset,
        has_more=has_more,
    )


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
    require_staff()
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
    require_staff()
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


@router.put("/{order_id}/owner", response_model=OrderResponse)
async def assign_order_owner(order_id: str, payload: OrderOwnerUpdate, session: DbSession):
    require_staff()
    if payload.owner_id not in settings.CUSTOMER_ACCESS_KEYS:
        raise HTTPException(422, "UNKNOWN_OWNER")
    order = await session.scalar(select(Order).where(Order.order_id == order_id).with_for_update())
    if order is None:
        raise HTTPException(404, "ORDER_NOT_FOUND")
    if order.owner_id not in (None, payload.owner_id):
        raise HTTPException(409, "ORDER_ALREADY_ASSIGNED")
    order.owner_id = payload.owner_id
    await session.commit()
    await session.refresh(order)
    return OrderResponse.model_validate(order)
