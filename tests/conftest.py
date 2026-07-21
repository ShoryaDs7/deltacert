import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=os.environ.get("DELTACERT_SMOKE_MODEL", "D:/models/llama3-3b"),
        help="Path to local HuggingFace model directory for smoke tests",
    )


def pytest_configure(config):
    try:
        pytest.smoke_model = config.getoption("--model")
    except ValueError:
        pytest.smoke_model = "D:/models/llama3-3b"
