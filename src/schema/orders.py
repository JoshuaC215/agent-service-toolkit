from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class OrderResponse(BaseModel):
    """Response schema for an order."""

    model_config = ConfigDict(from_attributes=True)

    order_id: str = Field(max_length=32)
    status: str = Field(max_length=32)

    amount: Decimal = Field(
        ge=0,
        max_digits=10,
        decimal_places=2,
    )

    currency: str = Field(
        min_length=3,
        max_length=3,
    )

    created_at: datetime
    updated_at: datetime


class OrderCreate(BaseModel):
    """Request schema for creating an order."""

    order_id: str = Field(min_length=1, max_length=32)
    status: str = Field(min_length=1, max_length=32)

    amount: Decimal = Field(
        ge=0,
        max_digits=10,
        decimal_places=2,
    )

    currency: str = Field(
        min_length=3,
        max_length=3,
    )


class OrderStatusUpdate(BaseModel):
    """Request schema for updating an order status."""

    status: str = Field(min_length=1, max_length=32)


class OrderOwnerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_id: str = Field(min_length=1, max_length=64)


class MyOrdersQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class MyOrdersResponse(BaseModel):
    orders: list[OrderResponse]
    limit: int
    offset: int
    has_more: bool
