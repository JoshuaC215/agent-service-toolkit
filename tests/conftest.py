import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-local-rag",
        action="store_true",
        default=False,
        help="run retrieval tests with the downloaded local BGE model",
    )
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")
    config.addinivalue_line("markers", "local_rag: requires downloaded local BGE weights")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-local-rag"):
        skip_local_rag = pytest.mark.skip(reason="need --run-local-rag option to run")
        for item in items:
            if "local_rag" in item.keywords:
                item.add_marker(skip_local_rag)
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Clear application configuration while preserving OS runtime variables."""
    system_keys = (
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
    )
    system_env = {key: os.environ[key] for key in system_keys if key in os.environ}
    with patch.dict(os.environ, system_env, clear=True):
        yield
