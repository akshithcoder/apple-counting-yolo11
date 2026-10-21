import math

import pytest

from apple_counter.metrics import compute_count_metrics


def test_count_metrics_for_known_errors() -> None:
    metrics = compute_count_metrics([10, 20, 30], [12, 18, 35])

    assert metrics.images == 3
    assert metrics.mae == pytest.approx(3.0)
    assert metrics.rmse == pytest.approx(math.sqrt(11))
    assert metrics.bias == pytest.approx(5 / 3)
    assert metrics.mean_true == pytest.approx(20)
    assert metrics.mean_predicted == pytest.approx(65 / 3)


def test_perfect_linear_counts_have_unit_correlation() -> None:
    metrics = compute_count_metrics([1, 2, 3], [2, 4, 6])

    assert metrics.correlation == pytest.approx(1.0)


def test_count_metrics_require_aligned_nonempty_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_count_metrics([1], [1, 2])
    with pytest.raises(ValueError, match="At least one"):
        compute_count_metrics([], [])
