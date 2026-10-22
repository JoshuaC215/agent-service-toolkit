"""Check startup configuration without opening sockets or connecting to a database."""

import asyncio
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import uvicorn

from core import settings


@pytest.mark.parametrize(
    "platform,expected_loop",
    [
        ("win32", "asyncio:SelectorEventLoop"),
        ("linux", "auto"),
    ],
)
def test_service_startup_loop_factory(monkeypatch, platform, expected_loop):
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(settings, "MODE", None)
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "src/run_service.py"), run_name="__main__"
    )
    assert run.call_args.kwargs["loop"] == expected_loop
    if platform == "win32":
        config = uvicorn.Config("service:app", **run.call_args.kwargs)
        assert config.get_loop_factory() is asyncio.SelectorEventLoop
