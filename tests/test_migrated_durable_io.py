"""Regression tests owned by the public acquisition package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import congreso_open_data.durable_io as durable_io
from congreso_open_data.durable_io import append_jsonl_durably, write_json_atomically


def test_durable_json_replace_and_jsonl_append_round_trip(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    events = tmp_path / "events.jsonl"

    write_json_atomically(state, {"status": "running", "completed": 1})
    write_json_atomically(state, {"status": "completed", "completed": 2})
    append_jsonl_durably(events, {"event": "first"})
    append_jsonl_durably(events, {"event": "second"})

    assert json.loads(state.read_text(encoding="utf-8")) == {
        "status": "completed",
        "completed": 2,
    }
    assert [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()] == [
        {"event": "first"},
        {"event": "second"},
    ]
    assert not list(tmp_path.glob(".*.tmp"))


def test_durable_json_fsync_failure_preserves_previous_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state.json"
    write_json_atomically(state, {"status": "previous"})

    def fail_fsync(_: int) -> None:
        raise OSError("forced fsync failure")

    monkeypatch.setattr(durable_io.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="forced fsync failure"):
        write_json_atomically(state, {"status": "new"})

    assert json.loads(state.read_text(encoding="utf-8")) == {"status": "previous"}
    assert not list(tmp_path.glob(".*.tmp"))


def test_durable_json_retries_transient_replace_sharing_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state.json"
    write_json_atomically(state, {"status": "previous"})
    real_replace = durable_io.os.replace
    attempts = 0
    delays: list[float] = []

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated Windows sharing violation")
        real_replace(source, target)

    monkeypatch.setattr(durable_io.os, "replace", flaky_replace)
    monkeypatch.setattr(durable_io.time, "sleep", delays.append)

    write_json_atomically(state, {"status": "completed"})

    assert attempts == 3
    assert delays == [0.025, 0.05]
    assert json.loads(state.read_text(encoding="utf-8")) == {"status": "completed"}
    assert not list(tmp_path.glob(".*.tmp"))


def test_durable_json_exhausted_replace_retries_preserve_previous_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state.json"
    write_json_atomically(state, {"status": "previous"})

    def locked_replace(_source: Path, _target: Path) -> None:
        raise PermissionError("permanently locked")

    monkeypatch.setattr(durable_io.os, "replace", locked_replace)
    monkeypatch.setattr(durable_io.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError, match="permanently locked"):
        write_json_atomically(state, {"status": "new"})

    assert json.loads(state.read_text(encoding="utf-8")) == {"status": "previous"}
    assert not list(tmp_path.glob(".*.tmp"))
