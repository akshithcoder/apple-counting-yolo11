"""Small helpers for YOLO label files."""

from pathlib import Path


def count_yolo_annotations(path: Path) -> int:
    """Count nonblank annotation rows, with an actionable error for a missing label."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
