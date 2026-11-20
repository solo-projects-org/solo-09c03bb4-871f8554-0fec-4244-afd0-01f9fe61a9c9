"""Shared test fixtures: isolated DATA_DIR and FastAPI test client."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("UPLOAD_TMP_DIR", str(data_dir / "tmp"))
    monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "128MB")
    # Import lazily so env vars are read at init time.
    from app import db as db_mod
    from app.main import app
    from fastapi.testclient import TestClient

    db_mod.init_db()
    with TestClient(app) as client:
        yield client, data_dir
    db_mod._writer.close()
    db_mod._writer = None


@pytest.fixture()
def paths(tmp_path):
    from tests.factories import hand_built

    return hand_built(tmp_path)
