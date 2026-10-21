#!/usr/bin/env python3
"""Report capture-session overlap in the existing train/validation split."""

import argparse
import json
from pathlib import Path

from apple_counter.splits import count_capture_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/yolo"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_counts = count_capture_sessions(args.data_root / "images/train")
    validation_counts = count_capture_sessions(args.data_root / "images/val")
    shared = sorted(set(train_counts) & set(validation_counts))
    result = {
        "session_definition": "filename prefix before _imageNNN",
        "train_images": sum(train_counts.values()),
        "validation_images": sum(validation_counts.values()),
        "train_session_count": len(train_counts),
        "validation_session_count": len(validation_counts),
        "shared_session_count": len(shared),
        "shared_sessions": shared,
        "train_images_by_session": train_counts,
        "validation_images_by_session": validation_counts,
    }
    serialized = json.dumps(result, indent=2) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
