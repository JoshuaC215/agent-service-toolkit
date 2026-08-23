"""Invariants for the model catalog that get_model's dispatch depends on."""

import typing
from collections import defaultdict

import pytest

from core.llm import _MODEL_TABLE
from schema.models import AllModelEnum, GoogleModelName, VertexAIModelName

MODEL_ENUMS = typing.get_args(AllModelEnum.__value__)


def test_model_enums_are_all_discovered():
    # An empty tuple would make the tests below pass without checking anything.
    assert len(MODEL_ENUMS) >= 12


def test_model_values_are_unique_across_enums():
    # StrEnum hashes by value, so members sharing a value collapse into one entry in
    # _MODEL_TABLE and AVAILABLE_MODELS, and get_model routes both to the first branch.
    owners = defaultdict(list)
    for enum in MODEL_ENUMS:
        for member in enum:
            owners[member.value].append(f"{enum.__name__}.{member.name}")

    duplicates = {value: names for value, names in owners.items() if len(names) > 1}
    assert not duplicates, f"model values must be unique across enums: {duplicates}"


@pytest.mark.parametrize("model", list(VertexAIModelName), ids=lambda m: m.value)
def test_vertexai_models_do_not_collide_with_google(model):
    # Every Vertex model has a Google twin, so a collision here builds the wrong client.
    assert model not in GoogleModelName


def test_model_table_covers_every_model():
    # _MODEL_TABLE is maintained separately from the AllModelEnum union.
    assert set(_MODEL_TABLE) == {member for enum in MODEL_ENUMS for member in enum}
