"""
Shared test configuration.

Sets SELDON_DB_PATH to a temporary file for each test session to
prevent SQLite database accumulation and cross-test contamination.
"""

import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_db(tmp_path_factory):
    """Use a temp DB path for all tests to avoid polluting the project dir."""
    tmp_dir = tmp_path_factory.mktemp("seldon_test_data")
    db_path = str(tmp_dir / "test_seldon.db")
    os.environ["SELDON_DB_PATH"] = db_path
    yield
    os.environ.pop("SELDON_DB_PATH", None)
