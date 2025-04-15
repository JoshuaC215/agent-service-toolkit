from typing import Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.types import StreamWriter

from agents.utils import CustomData
from schema.task_data import TaskData


class Task:
    def __init__(self, task_name: str, writer: StreamWriter | None = None) -> None:
        self.name = task_name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Literal["success", "error"] | None = None
        self.writer = writer

    def _generate_and_dispatch_message(self, writer: StreamWriter | None, data: dict):
        writer = writer or self.writer
        task_data = TaskData(name=self.name, run_id=self.id, state=self.state, data=data)
        if self.result:
            task_data.result = self.result
        task_custom_data = CustomData(
            type=self.name,
            data=task_data.model_dump(),
        )
        if writer:
            task_custom_data.dispatch(writer)
        return task_custom_data.to_langchain()

    def start(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        self.state = "new"
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message

    def write_data(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        if self.state == "complete":
            raise ValueError("Only incomplete tasks can output data.")
        self.state = "running"
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message

    def finish(
        self,
        result: Literal["success", "error"],
        writer: StreamWriter | None = None,
        data: dict = {},
    ) -> BaseMessage:
        self.state = "complete"
        self.result = result
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message
