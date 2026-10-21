#!/usr/bin/env python3
"""Convert MinneApple instance annotations into a deterministic YOLO split."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from apple_counter.annotations import Box, mask_to_boxes, to_yolo_line

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minneapple-root",
        type=Path,
        default=Path("data/minneapple"),
        help="Directory containing extracted MinneApple detection and test_data folders.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def prepare_split(
    names: list[str],
    split: str,
    boxes_by_name: dict[str, list[Box]],
    image_directory: Path,
    output: Path,
) -> int:
    apple_count = 0
    for name in names:
        image_path = image_directory / name
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        lines = [
            to_yolo_line(box, image_width, image_height)
            for box in boxes_by_name[name]
        ]
        write_label(output / "labels" / split / f"{Path(name).stem}.txt", lines)
        link_image(image_path, output / "images" / split / name)
        apple_count += len(lines)
    return apple_count


def validate_source_layout(root: Path) -> tuple[Path, Path, Path, Path]:
    train_images = root / "detection/train/images"
    train_masks = root / "detection/train/masks"
    test_images = root / "detection/test/images"
    test_ground_truth = root / "test_data/detection/ground_truth.json"
    required = (train_images, train_masks, test_images, test_ground_truth)
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing MinneApple inputs:\n{formatted}")
    return required


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output} already exists; pass --overwrite to replace it."
            )
        shutil.rmtree(args.output)

    train_images, train_masks, test_images, test_ground_truth = validate_source_layout(
        args.minneapple_root
    )
    names = sorted(path.name for path in train_images.glob("*.png"))
    mask_names = {path.name for path in train_masks.glob("*.png")}
    if not names:
        raise ValueError(f"No PNG training images found in {train_images}.")
    if set(names) != mask_names:
        raise ValueError("Every training image must have a mask with the same filename.")

    boxes_by_name = {
        name: mask_to_boxes(np.asarray(Image.open(train_masks / name))) for name in names
    }
    random.Random(args.seed).shuffle(names)
    validation_size = round(len(names) * args.val_fraction)
    validation_names = sorted(names[:validation_size])
    training_names = sorted(names[validation_size:])

    training_apples = prepare_split(
        training_names, "train", boxes_by_name, train_images, args.output
    )
    validation_apples = prepare_split(
        validation_names, "val", boxes_by_name, train_images, args.output
    )

    ground_truth = json.loads(test_ground_truth.read_text(encoding="utf-8"))
    test_boxes: dict[str, list[Box]] = {
        image["filename"]: [] for image in ground_truth["images"]
    }
    image_id_to_name = {
        image["id"]: image["filename"] for image in ground_truth["images"]
    }
    for annotation in ground_truth["annotations"]:
        x, y, width, height = annotation["bbox"]
        name = image_id_to_name[annotation["image_id"]]
        # COCO boxes may contain fractional coordinates; preserve them during normalization.
        test_boxes[name].append((float(x), float(y), float(width), float(height)))
    if not test_boxes:
        raise ValueError(f"No test images found in {test_ground_truth}.")
    test_names = sorted(test_boxes)
    test_apples = prepare_split(
        test_names, "test", test_boxes, test_images, args.output
    )

    dataset_yaml = args.output / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {args.output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: apple",
                "",
            ]
        ),
        encoding="utf-8",
    )
    LOGGER.info("train: %d images, %d apples", len(training_names), training_apples)
    LOGGER.info("validation: %d images, %d apples", len(validation_names), validation_apples)
    LOGGER.info("test: %d images, %d apples", len(test_names), test_apples)
    LOGGER.info("dataset configuration: %s", dataset_yaml)


if __name__ == "__main__":
    main()
