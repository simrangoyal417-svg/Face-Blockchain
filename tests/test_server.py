"""Tests for the web server API endpoints."""

import base64
import json
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest
import requests

from app import AppRequestHandler, PROJECT_ROOT


@pytest.fixture(scope="module")
def test_server():
    """Start the test server on an ephemeral port in a background thread."""
    server = HTTPServer(("127.0.0.1", 0), AppRequestHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_server_get_index(test_server: str):
    res = requests.get(f"{test_server}/")
    assert res.status_code == 200
    assert "Face Identification" in res.text


def test_server_get_candidates(test_server: str):
    res = requests.get(f"{test_server}/api/candidates")
    assert res.status_code == 200
    data = res.json()
    assert "candidates" in data
    assert len(data["candidates"]) > 0


def test_server_get_ledger(test_server: str):
    res = requests.get(f"{test_server}/api/ledger")
    assert res.status_code == 200
    data = res.json()
    assert "blocks" in data


def test_server_post_scan_image_path(test_server: str):
    res = requests.post(
        f"{test_server}/api/scan",
        json={"image_path": "input/sample.png", "threshold": 0.36},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["face_detected"] is True
    assert data["is_match"] is True
    assert data["blockchain_block"] is not None
    assert data["blockchain_verification"]["verified"] is True


def test_server_post_scan_base64(test_server: str):
    sample_bytes = (PROJECT_ROOT / "input" / "sample.png").read_bytes()
    b64_str = f"data:image/png;base64,{base64.b64encode(sample_bytes).decode('utf-8')}"
    res = requests.post(
        f"{test_server}/api/scan",
        json={"image_data": b64_str, "threshold": 0.36},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["face_detected"] is True
    assert data["is_match"] is True
    assert data["blockchain_block"] is not None
