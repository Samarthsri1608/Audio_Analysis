"""
conftest.py — pytest configuration for the AI Speaking Assessment test suite.
Adds a --base-url CLI option so you can target a different host if needed.
"""
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running FastAPI server (default: http://localhost:8000)",
    )

def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown marks."""
    config.addinivalue_line("markers", "slow: marks tests as slow (use -m 'not slow' to skip)")
