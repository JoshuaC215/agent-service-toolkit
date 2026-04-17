import json
import re
import unicodedata
from os import path as ospath
from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field

# Simple in-process caches to reduce I/O
_PROMPT_CACHE: dict[str, str] = {}
_QUESTIONS_CACHE: dict[str, Any] = {}


class CustomData(BaseModel):
    "Custom data being sent by an agent"

    data: dict[str, Any] = Field(description="The custom data")

    def to_langchain(self) -> ChatMessage:
        # LangChain ChatMessage expects content to be a string or a list; wrap dict in a list
        return ChatMessage(content=[self.data], role="custom")

    def dispatch(self, writer: StreamWriter) -> None:
        writer(self.to_langchain())


def load_questions(filename: str = "skill_questions.json"):
    """
    Load questions from a JSON file in the agents_questions directory by filename only.
    Guards against path traversal and caches reads to reduce I/O.
    """
    # Enforce filename-only to prevent path traversal
    if (
        filename != ospath.basename(filename)
        or ".." in filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("Invalid filename; only plain filenames are allowed.")
    # Cache
    if filename in _QUESTIONS_CACHE:
        return _QUESTIONS_CACHE[filename]
    base_dir = ospath.join(ospath.dirname(__file__), "agents_questions")
    path = ospath.join(base_dir, filename)
    try:
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)
        # Validate structure
        if not isinstance(questions, list):
            raise ValueError("Questions JSON must be a list.")
        for q in questions:
            if not isinstance(q, dict) or "text" not in q or "type" not in q:
                raise ValueError("Each question must be a dict with 'text' and 'type'.")
        _QUESTIONS_CACHE[filename] = questions
        return questions
    except Exception as e:
        raise RuntimeError(f"Failed to load questions from {path}: {e}")


def load_prompt(filename: str) -> str:
    """
    Load a prompt template from a text file in the prompts directory by filename only.
    Guards against path traversal and caches reads to reduce I/O.
    """
    # Enforce filename-only to prevent path traversal
    if (
        filename != ospath.basename(filename)
        or ".." in filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("Invalid filename; only plain filenames are allowed.")
    # Cache
    if filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]
    base_dir = ospath.join(ospath.dirname(__file__), "prompts")
    path = ospath.join(base_dir, filename)
    try:
        with open(path, encoding="utf-8") as f:
            template = f.read()
        _PROMPT_CACHE[filename] = template
        return template
    except Exception as e:
        raise RuntimeError(f"Failed to load prompt from {path}: {e}")


class JSONCleaner:
    """
    Cleans messy or slightly invalid JSON returned by LLMs and ensures it's valid JSON.
    Uses json_repair under the hood and applies optional sanitization steps.
    """

    def clean_json(self, json_str: str, return_dict: bool = False) -> Any:
        """
        Clean the input JSON string: slice outer braces, sanitize, repair common errors,
        then validate by parsing. Returns either a parsed dict (when return_dict=True)
        or a UTF-8 JSON string via json.dumps(..., ensure_ascii=False).
        """
        try:
            from json_repair import repair_json
        except ImportError as e:
            msg = "Could not import the json_repair package. Please install it with `uv add json-repair`."
            raise ImportError(msg) from e

        start = json_str.find("{")
        end = json_str.rfind("}")
        if start == -1 or end == -1:
            msg = "Invalid JSON string: Missing '{' or '}'"
            raise ValueError(msg)
        try:
            sliced = json_str[start : end + 1]
            # Basic sanitization before repair
            sanitized = self._remove_control_characters(sliced)
            sanitized = self._normalize_unicode(sanitized)
            # Repair common issues first
            repaired = repair_json(sanitized)
            # Validate by parsing
            parsed = json.loads(repaired)
            parsed = self._sanitize_parsed(parsed)
            return parsed if return_dict else json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            msg = f"Error cleaning JSON string: {e}"
            raise ValueError(msg) from e

    def _remove_control_characters(self, s: str) -> str:
        return re.sub(r"[\x00-\x1F\x7F]", "", s)

    def _normalize_unicode(self, s: str) -> str:
        return unicodedata.normalize("NFC", s)

    def _sanitize_parsed(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._sanitize_parsed(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_parsed(v) for v in value]
        if isinstance(value, str):
            return self._normalize_unicode(self._remove_control_characters(value))
        return value

    def _validate_json(self, s: str) -> None:
        try:
            json.loads(s)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON string: {e}"
            raise ValueError(msg) from e


def current_date_str(fmt: str = "%B %d, %Y") -> str:
    """
    Convenience helper to format current date for instruction strings.
    """
    from datetime import datetime

    return datetime.now().strftime(fmt)
