from pathlib import Path

import pytest

from apple_counter.labels import count_yolo_annotations


def test_annotation_count_ignores_blank_and_whitespace_rows(tmp_path: Path) -> None:
    label = tmp_path / "image.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n\n   \n0 0.2 0.3 0.1 0.2\n")

    assert count_yolo_annotations(label) == 2


def test_missing_annotation_has_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Annotation file not found"):
        count_yolo_annotations(missing)


def test_empty_annotation_file_represents_zero_instances(tmp_path: Path) -> None:
    label = tmp_path / "empty.txt"
    label.write_text("")

    assert count_yolo_annotations(label) == 0
