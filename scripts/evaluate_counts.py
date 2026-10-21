#!/usr/bin/env python3
"""Evaluate image-level apple counts on validation and test splits."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

from apple_counter.labels import count_yolo_annotations
from apple_counter.metrics import CountMetrics, compute_count_metrics

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/apple-yolo11n-best.pt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/yolo"))
    parser.add_argument("--output", type=Path, default=Path("runs/count-evaluation"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def evaluate_split(
    model: YOLO,
    split: str,
    data_root: Path,
    confidence: float,
    nms_iou: float,
    max_detections: int,
    image_size: int,
    device: str | None,
) -> list[dict[str, str | int]]:
    image_directory = data_root / "images" / split
    label_directory = data_root / "labels" / split
    if not image_directory.exists() or not label_directory.exists():
        raise FileNotFoundError(f"Missing images or labels for split '{split}'.")
    if not any(image_directory.glob("*.png")):
        raise ValueError(f"No PNG images found for split '{split}' in {image_directory}.")

    predict_options = {
        "source": str(image_directory),
        "imgsz": image_size,
        "conf": confidence,
        "iou": nms_iou,
        "max_det": max_detections,
        "stream": True,
        "verbose": False,
    }
    if device is not None:
        predict_options["device"] = device

    rows: list[dict[str, str | int]] = []
    for result in model.predict(**predict_options):
        image_path = Path(result.path)
        true_count = count_yolo_annotations(
            label_directory / f"{image_path.stem}.txt"
        )
        predicted_count = 0 if result.boxes is None else len(result.boxes)
        rows.append(
            {
                "split": split,
                "photo": image_path.name,
                "true_count": true_count,
                "predicted_count": predicted_count,
                "error": predicted_count - true_count,
            }
        )
    return rows


def summarize(rows: list[dict[str, str | int]]) -> CountMetrics:
    return compute_count_metrics(
        [int(row["true_count"]) for row in rows],
        [int(row["predicted_count"]) for row in rows],
    )


def save_scatter(rows: list[dict[str, str | int]], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    for split, color in (("val", "#1f77b4"), ("test", "#ff7f0e")):
        split_rows = [row for row in rows if row["split"] == split]
        axis.scatter(
            [row["true_count"] for row in split_rows],
            [row["predicted_count"] for row in split_rows],
            s=18,
            alpha=0.7,
            label=split,
            color=color,
        )
    limit = max(
        max(int(row["true_count"]), int(row["predicted_count"])) for row in rows
    ) + 5
    axis.plot([0, limit], [0, limit], color="black", linewidth=1)
    axis.set_xlabel("Ground-truth apple count")
    axis.set_ylabel("Predicted apple count")
    axis.set_title("Image-level apple counts")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    rows = [
        *evaluate_split(
            model,
            "val",
            args.data_root,
            args.confidence,
            args.nms_iou,
            args.max_detections,
            args.image_size,
            args.device,
        ),
        *evaluate_split(
            model,
            "test",
            args.data_root,
            args.confidence,
            args.nms_iou,
            args.max_detections,
            args.image_size,
            args.device,
        ),
    ]

    predictions_path = args.output / "count_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = {}
    for split in ("val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        summaries[split] = summarize(split_rows).to_dict()
        LOGGER.info("%s: %s", split, summaries[split])
    summaries["settings"] = {
        "confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "max_detections": args.max_detections,
        "image_size": args.image_size,
        "weights": str(args.weights),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    save_scatter(rows, args.output / "count_scatter.png")


if __name__ == "__main__":
    main()
