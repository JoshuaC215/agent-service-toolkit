import json
import os
from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field


class CustomData(BaseModel):
    "Custom data being sent by an agent"

    data: dict[str, Any] = Field(description="The custom data")

    def to_langchain(self) -> ChatMessage:
        return ChatMessage(content=[self.data], role="custom")

    def dispatch(self, writer: StreamWriter) -> None:
        writer(self.to_langchain())


def load_questions(filename="skill_questions.json"):
    """
    Load questions from a JSON file in the agents_questions directory.
    Default file: skill_questions.json
    """
    base_dir = os.path.join(os.path.dirname(__file__), "agents_questions")
    path = os.path.join(base_dir, filename)
    try:
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)
        # Validate structure
        if not isinstance(questions, list):
            raise ValueError("Questions JSON must be a list.")
        for q in questions:
            if not isinstance(q, dict) or "text" not in q or "type" not in q:
                raise ValueError("Each question must be a dict with 'text' and 'type'.")
        return questions
    except Exception as e:
        raise RuntimeError(f"Failed to load questions from {path}: {e}")


def load_prompt(filename: str, **kwargs) -> str:
    """
    Load a prompt template from a text file in the prompts directory and format it with kwargs.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "prompts")
    path = os.path.join(base_dir, filename)
    try:
        with open(path, encoding="utf-8") as f:
            template = f.read()
        if kwargs:
            return template.format(**kwargs)
        return template
    except Exception as e:
        raise RuntimeError(f"Failed to load prompt from {path}: {e}")
