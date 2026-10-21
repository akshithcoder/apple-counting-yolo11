# Local data layout

Dataset files are intentionally excluded from version control. Download MinneApple from the
[University of Minnesota Data Repository](https://doi.org/10.13020/8ecp-3r13), then arrange the
two extracted archives as follows:

```text
data/minneapple/
├── detection/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/images/
└── test_data/detection/ground_truth.json
```

The original archives used in this experiment extract into this layout with:

```bash
tar -xzf detection.tar.gz -C data/minneapple
unzip test_data.zip -d data/minneapple
```

Run `python scripts/prepare_data.py` afterward. It creates `data/yolo/` with deterministic
train/validation/test splits and a machine-local `dataset.yaml`.

MinneApple is distributed under CC BY-NC-SA 3.0 US. Do not commit the downloaded data to this
repository; review the source terms before using or redistributing it.
