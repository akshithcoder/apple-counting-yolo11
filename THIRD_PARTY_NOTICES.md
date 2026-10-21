# Third-party notices

## Ultralytics YOLO11

This project uses the `ultralytics` package and a YOLO11n pretrained checkpoint. Ultralytics
states that its open-source software and YOLO11 models are available under the GNU Affero
General Public License v3.0, with a separate enterprise-license option. The repository code and
included fine-tuned checkpoint are therefore distributed under AGPL-3.0-only. See the
[Ultralytics licensing page](https://www.ultralytics.com/license) and
[YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

Ultralytics citation:

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## MinneApple

The experiment was trained and evaluated on MinneApple. The raw dataset is not included. The
files under `assets/predictions/` contain MinneApple images with model annotations and are
redistributed for non-commercial research and documentation under CC BY-NC-SA 3.0 US, with the
same attribution as the source dataset. Figures under `assets/figures/` and experiment tables
were derived from the same research run.

Dataset citation:

```bibtex
@misc{hani2019minneapple,
  title = {MinneApple: A Benchmark Dataset for Apple Detection and Segmentation},
  author = {Nicolai Häni and Pravakar Roy and Volkan Isler},
  year = {2019},
  eprint = {1909.06441},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.13020/8ecp-3r13}
}
```

The upstream MinneApple code repository is MIT-licensed. Dataset media have separate terms; see
the [dataset record](https://doi.org/10.13020/8ecp-3r13) before reuse. Because the included model
was trained on non-commercially licensed data, downstream users should independently assess the
dataset terms for their intended use.
