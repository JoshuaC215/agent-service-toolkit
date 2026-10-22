from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TicketResponse(BaseModel):
    """Response schema for a support ticket."""

    model_config = ConfigDict(from_attributes=True)

    ticket_id: str = Field(max_length=32)
    order_id: str = Field(max_length=32)
    issue_type: str = Field(max_length=32)
    description: str
    status: str = Field(max_length=32)
    created_at: datetime
    updated_at: datetime


class TicketCreate(BaseModel):
    """Request schema for creating a support ticket."""

    ticket_id: str = Field(min_length=1, max_length=32)
    order_id: str = Field(min_length=1, max_length=32)
    issue_type: str = Field(min_length=1, max_length=32)
    description: str = Field(min_length=1)


class TicketStatusUpdate(BaseModel):
    """Request schema for updating a ticket status."""

    status: str = Field(min_length=1, max_length=32)