from pathlib import Path

import pytest

from apple_counter.splits import capture_session, count_capture_sessions


def test_capture_session_uses_timestamp_prefix() -> None:
    assert capture_session("20150921_131625_image646.png") == "20150921_131625"


def test_capture_session_rejects_unrecognized_name() -> None:
    with pytest.raises(ValueError, match="Unrecognized"):
        capture_session("dataset1_back_1021.png")


def test_count_capture_sessions_rejects_missing_and_empty_directories(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        count_capture_sessions(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No PNG images"):
        count_capture_sessions(empty)
