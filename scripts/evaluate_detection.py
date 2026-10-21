#!/usr/bin/env python3
"""Evaluate the saved detector on the independent MinneApple test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/apple-yolo11n-best.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/yolo/dataset.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/test-detection/yolo11n-960"),
    )
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")
    if args.output.exists():
        if not args.output.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    # These are Ultralytics validation defaults, not the 0.25 counting threshold.
    confidence_threshold = 0.001
    nms_iou_threshold = 0.7
    maximum_detections = 300
    settings = {
        "weights": str(args.weights),
        "weights_sha256": sha256(args.weights),
        "data": str(args.data),
        "split": "test",
        "image_size": args.image_size,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "maximum_detections": maximum_detections,
        "metric_iou_thresholds": "0.50:0.05:0.95",
        "rectangular_batches": True,
        "augmentation": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    (args.output / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    log_handler = logging.FileHandler(args.output / "evaluation.log", mode="w")
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    ULTRALYTICS_LOGGER.addHandler(log_handler)
    try:
        model = YOLO(args.weights)
        metrics = model.val(
            data=str(args.data),
            split="test",
            imgsz=args.image_size,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            conf=confidence_threshold,
            iou=nms_iou_threshold,
            max_det=maximum_detections,
            rect=True,
            augment=False,
            plots=True,
            project=str(args.output.parent.resolve()),
            name=args.output.name,
            exist_ok=True,
            verbose=True,
        )
    finally:
        ULTRALYTICS_LOGGER.removeHandler(log_handler)
        log_handler.close()

    results = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map_50": float(metrics.box.map50),
        "map_50_95": float(metrics.box.map),
        "fitness": float(metrics.fitness),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
