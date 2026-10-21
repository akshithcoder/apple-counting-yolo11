import numpy as np
import pytest

from apple_counter.annotations import mask_to_boxes, to_yolo_coordinates, to_yolo_line


def test_mask_to_boxes_uses_nonzero_instance_ids() -> None:
    mask = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 2],
        ],
        dtype=np.uint8,
    )

    assert mask_to_boxes(mask) == [(1, 0, 2, 2), (4, 1, 1, 2)]


def test_mask_to_boxes_rejects_color_image() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mask_to_boxes(np.zeros((2, 2, 3), dtype=np.uint8))


def test_pixel_box_is_normalized_to_yolo_coordinates() -> None:
    coordinates = to_yolo_coordinates((20, 10, 40, 20), 100, 50)

    assert coordinates == pytest.approx((0.4, 0.4, 0.4, 0.4))
    assert to_yolo_line((20, 10, 40, 20), 100, 50) == (
        "0 0.400000 0.400000 0.400000 0.400000"
    )


def test_out_of_bounds_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        to_yolo_coordinates((90, 10, 20, 20), 100, 50)


def test_fractional_coco_box_is_not_truncated() -> None:
    coordinates = to_yolo_coordinates((0.5, 1.25, 2.5, 3.5), 10, 10)

    assert coordinates == pytest.approx((0.175, 0.3, 0.25, 0.35))
    assert to_yolo_line((0.5, 1.25, 2.5, 3.5), 10, 10) == (
        "0 0.175000 0.300000 0.250000 0.350000"
    )
