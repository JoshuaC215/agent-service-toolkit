from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    execution_id: str
    approval_id: str
    order_id: str
    amount: Decimal
    currency: str
    status: Literal["simulated_succeeded"]
    executor_id: str
    result_message: str
    executed_at: datetime


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_simulation: Literal[True]


class SimulationResponse(BaseModel):
    execution: ExecutionResponse
    created: bool
    simulated: Literal[True] = True
    funds_transferred: Literal[False] = False
