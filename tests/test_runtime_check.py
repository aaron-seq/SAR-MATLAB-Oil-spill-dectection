"""The HTTP harness retries startup failures without hiding accepted-job errors."""

from http.client import RemoteDisconnected
from unittest.mock import Mock
from urllib.error import URLError

import pytest
from scripts import verify_batch_runtime as runtime


@pytest.mark.parametrize("error", [
    ConnectionResetError("starting"), ConnectionRefusedError("starting"),
    RemoteDisconnected("starting"), TimeoutError("starting"), URLError("starting"),
], ids=["reset", "refused", "remote_disconnect", "timeout", "url_error"])
def test_readiness_waits_for_ready_after_a_startup_connection_error(monkeypatch, error):
    transport = Mock(side_effect=[error, (200, {"detector_ready": True})])
    monkeypatch.setattr(runtime, "request", transport)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)

    runtime.wait_ready("http://127.0.0.1:8000")

    assert transport.call_count == 2  # The ready response must actually be reached.


def test_readiness_stops_at_its_deadline_despite_startup_resets(monkeypatch):
    monkeypatch.setattr(runtime, "request", Mock(side_effect=ConnectionResetError("starting")))
    monkeypatch.setattr(runtime.time, "monotonic", Mock(side_effect=[10, 10, 189.9, 190]))
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="did not become ready within 180 seconds"):
        runtime.wait_ready("http://127.0.0.1:8000")


@pytest.mark.parametrize("allow_missing", [False, True])
def test_connection_reset_after_job_acceptance_is_not_retried(monkeypatch, allow_missing):
    transport = Mock(side_effect=[
        (200, {"batch_id": "accepted-job"}), ConnectionResetError("poll interrupted"),
    ])
    monkeypatch.setattr(runtime, "request", transport)

    with pytest.raises(ConnectionResetError, match="poll interrupted"):
        runtime.job("http://127.0.0.1:8000", b"upload", 1, allow_missing)
