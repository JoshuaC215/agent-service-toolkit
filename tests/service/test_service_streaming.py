import pytest
from langchain_core.messages import AIMessage

from service.service import _create_ai_message


@pytest.mark.parametrize(
    "parts, expected",
    [
        # 1) Basic content + tool_calls
        (
            {"content": "Hello", "tool_calls": []},
            {"content": "Hello", "tool_calls": []},
        ),
        # 2) Unknown keys are ignored
        (
            {"content": "Test", "foobar": 123, "tool_calls": []},
            {"content": "Test", "tool_calls": []},
        ),
        # 3) Extra valid AIMessage params (id, type) pass through
        (
            {
                "content": "Hey",
                "id": "abc-123",
                "type": "ai",
                "tool_calls": [],
            },
            {"content": "Hey", "id": "abc-123", "type": "ai", "tool_calls": []},
        ),
    ],
)
def test_create_ai_message_filters_and_passes_through(parts, expected):
    """
    _create_ai_message should:
      - Drop unknown keys ("foobar")
      - Preserve keys that match AIMessage signature
      - Use the final value for duplicate keys in the parts dict
    """
    msg: AIMessage = _create_ai_message(parts)
    for key, val in expected.items():
        assert getattr(msg, key) == val


def test_create_ai_message_missing_required_content_raises():
    """
    AIMessage requires 'content'; if missing, _create_ai_message should
    bubble up the TypeError from the constructor.
    """
    with pytest.raises(TypeError):
        _create_ai_message({"tool_calls": []})


def test_create_ai_message_empty_dict_raises():
    """
    Completely empty parts should also fail to construct an AIMessage.
    """
    with pytest.raises(TypeError):
        _create_ai_message({})
