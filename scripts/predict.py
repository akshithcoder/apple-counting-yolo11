#!/usr/bin/env python3
"""Run apple detection on one image or a directory of images."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Image file or directory of images.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/apple-yolo11n-best.pt"),
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Prediction source not found: {args.source}")
    if args.source.is_dir() and not any(args.source.iterdir()):
        raise ValueError(f"Prediction source directory is empty: {args.source}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    predict_options = {
        "source": str(args.source),
        "conf": args.confidence,
        "iou": args.nms_iou,
        "max_det": args.max_detections,
        "imgsz": args.image_size,
        "save": True,
        # An absolute project path avoids interaction with user-level Ultralytics runs_dir settings.
        "project": str(args.output.parent.resolve()),
        "name": args.output.name,
        "exist_ok": True,
        "verbose": False,
    }
    if args.device is not None:
        predict_options["device"] = args.device

    model = YOLO(args.weights)
    results = model.predict(**predict_options)
    if not results:
        raise ValueError(f"No supported images found in prediction source: {args.source}")
    for result in results:
        count = 0 if result.boxes is None else len(result.boxes)
        print(f"{Path(result.path).name}: {count} apples")
    print(f"Annotated predictions saved to {results[0].save_dir}")


if __name__ == "__main__":
    main()
