# Model checkpoint

`apple-yolo11n-best.pt` is the best checkpoint saved by the verified 30-epoch run and is the
checkpoint used to produce the committed counting results.

- Architecture: YOLO11n detection, one class (`apple`)
- Parameters: 2,590,035
- File size: approximately 5.2 MB
- SHA-256: `b9710b229f9f0be979061b359f239fa2f35e36d4ee1d09691d58be1f44ca2807`

The checkpoint's machine-specific training-output path was replaced with `runs/detect/apple`
before publication. Learned tensors and predictions were not changed; all 465 saved validation
and test predictions were reproduced exactly after sanitization.

The checkpoint is distributed under AGPL-3.0-only in accordance with Ultralytics' open-source
model licensing. It was trained on MinneApple; consult `THIRD_PARTY_NOTICES.md` for the dataset's
separate non-commercial terms.
