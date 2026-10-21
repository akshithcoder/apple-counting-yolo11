"""Capture-session utilities for the deterministic MinneApple split."""

import re
from collections import Counter
from pathlib import Path

CAPTURE_SESSION_PATTERN = re.compile(r"^(?P<session>\d{8}_\d{6})_image\d+$")


def capture_session(filename: str) -> str:
    """Extract the timestamp prefix shared by images from one capture session."""
    match = CAPTURE_SESSION_PATTERN.fullmatch(Path(filename).stem)
    if match is None:
        raise ValueError(f"Unrecognized MinneApple training filename: {filename}")
    return match.group("session")


def count_capture_sessions(image_directory: Path) -> dict[str, int]:
    """Count PNG images per capture session in a nonempty split directory."""
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_directory}")
    image_paths = sorted(image_directory.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {image_directory}")
    return dict(Counter(capture_session(path.name) for path in image_paths))
