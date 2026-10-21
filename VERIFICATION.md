# Verification record

Executed on 2026-10-20 on macOS arm64 (Apple M4 Pro). This record describes commands that were
actually run; it does not imply that training was repeated.

## Passed

### Clean installation

A new Python 3.12.14 virtual environment was created outside the repository and installed with:

```bash
python3.12 -m venv /tmp/apple-counting-final-clean-20261020
/tmp/apple-counting-final-clean-20261020/bin/python -m pip install --upgrade pip
/tmp/apple-counting-final-clean-20261020/bin/python -m pip install ".[dev]"
```

The wheel built successfully and all pinned runtime and development dependencies installed.

### Checkpoint and inference

- `weights/apple-yolo11n-best.pt` loaded with Ultralytics 8.4.163.
- SHA-256: `b9710b229f9f0be979061b359f239fa2f35e36d4ee1d09691d58be1f44ca2807`.
- The loaded model had one class (`apple`) and 2,590,035 parameters.
- CPU inference ran on the real MinneApple test image `dataset1_back_1021.png` at image size 960,
  confidence 0.25, NMS IoU 0.7, and `max_det=300`.
- Result: 41 predicted apples; the committed ground-truth count is 41.

### Deterministic data preparation

The locally available MinneApple archives were processed into a separate output directory with
split seed 42 and validation fraction 0.20.

| Split | Images | Annotations |
|---|---:|---:|
| Train | 536 | 22,523 |
| Validation | 134 | 5,660 |
| Test | 331 | 12,285 |

All regenerated YOLO label files were byte-for-byte identical to the existing generated labels.
The generated image-link counts also matched for every split.

The original COCO test JSON contains 12,285 bounding boxes and 49,140 coordinate values. None
has a fractional component, so the former integer conversion caused no precision loss for this
dataset and changing the converter to preserve valid fractional COCO coordinates did not change
any generated label or historical metric.

### Counting evaluation

The saved checkpoint was rerun over all validation and test images with the original confidence
threshold 0.25 and the now-explicit historical defaults of NMS IoU 0.7 and `max_det=300`.

- All 465 per-image prediction rows matched the committed CSV exactly.
- Validation MAE 3.9403, RMSE 5.4087, bias +2.1940, correlation 0.9881.
- Test MAE 6.3565, RMSE 8.4888, bias +2.7855, correlation 0.9044.

### Independent test-set detection evaluation

The saved checkpoint was evaluated three times against the independent test bounding boxes. All
three runs produced identical detection metrics.

| Setting | Value |
|---|---|
| Images / instances | 331 / 12,285 |
| Image size / batch | 960 / 8 |
| Confidence | 0.001 (Ultralytics validation setting) |
| NMS IoU / maximum detections | 0.7 / 300 |
| AP matching thresholds | 0.50:0.05:0.95 |
| Device | Apple MPS |

| Metric | Result |
|---|---:|
| Precision | 0.803219 |
| Recall | 0.683435 |
| mAP@50 | 0.781004 |
| mAP@50–95 | 0.421902 |

Settings, full-precision metrics, and the evaluation log are stored in
`experiments/yolo11n_960/test_detection/`.

### Capture-session audit

Capture session was defined as the timestamp prefix before `_imageNNN` in the original training
filenames. Training contained 10 sessions, validation contained 10 sessions, and all 10 occurred
in both splits. This establishes session overlap and possible correlation, not duplicate-image or
label leakage.

### Static checks and tests

In the final clean Python 3.12 environment:

- `ruff check src scripts tests`: passed.
- `pytest -q`: 14 passed.
- All documented script `--help` entry points loaded successfully.
- The GitHub Actions `tests` workflow passed on the pushed review commit.

## Not run or not claimed

- The model was not retrained; historical weights and training results were preserved.
- Installation and inference were not tested on Linux, Windows, CUDA, or an NVIDIA GPU.
- The upstream dataset download itself was not repeated because the original archives were
  already available locally; extraction, conversion, and deterministic splitting were repeated.
- No new confidence, NMS, architecture, or training hyperparameters were selected using test
  results.
- The independent detection results are Ultralytics metrics over the local converted test labels,
  not a submission to the historical MinneApple CodaLab benchmark server.
