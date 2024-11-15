from typing import Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from agents.utils import CustomData
from schema.task_data import TaskData


class Task:
    def __init__(self, task_name: str) -> None:
        self.name = task_name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Literal["success", "error"] | None = None

    async def _generate_and_dispatch_message(self, config: RunnableConfig, data: dict):
        task_data = TaskData(name=self.name, run_id=self.id, state=self.state, data=data)
        if self.result:
            task_data.result = self.result
        task_custom_data = CustomData(
            type=self.name,
            data=task_data.model_dump(),
        )
        await task_custom_data.adispatch(config)
        return task_custom_data.to_langchain()

    async def start(self, config: RunnableConfig, data: dict = {}) -> BaseMessage:
        self.state = "new"
        task_message = await self._generate_and_dispatch_message(config, data)
        return task_message

    async def write_data(self, config: RunnableConfig, data: dict) -> BaseMessage:
        if self.state == "complete":
            raise ValueError("Only incomplete tasks can output data.")
        self.state = "running"
        task_message = await self._generate_and_dispatch_message(config, data)
        return task_message

    async def finish(
        self, result: Literal["success", "error"], config: RunnableConfig, data: dict = {}
    ) -> BaseMessage:
        self.state = "complete"
        self.result = result
        task_message = await self._generate_and_dispatch_message(config, data)
        return task_message
