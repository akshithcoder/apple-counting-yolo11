#!/usr/bin/env python3
"""Fine-tune YOLO11n using a YAML experiment configuration."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--data",
        type=Path,
        help="Override the dataset YAML path in the experiment configuration.",
    )
    parser.add_argument(
        "--device",
        help="Override the configured device, for example mps, cpu, or 0 for CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = configuration.pop("model")
    if args.data is not None:
        configuration["data"] = str(args.data)
    if args.device is not None:
        configuration["device"] = args.device

    # Resolve the project path so user-level Ultralytics settings cannot prepend another runs dir.
    configuration["project"] = str(Path(configuration["project"]).resolve())

    dataset_path = Path(configuration["data"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found at {dataset_path}. Run prepare_data.py first."
        )

    model = YOLO(model_name)
    model.train(**configuration)


if __name__ == "__main__":
    main()
