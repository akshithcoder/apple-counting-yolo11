"""Image-level apple-counting metrics."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import sqrt


@dataclass(frozen=True)
class CountMetrics:
    """Summary statistics for paired ground-truth and predicted counts."""

    images: int
    mae: float
    rmse: float
    bias: float
    mean_true: float
    mean_predicted: float
    correlation: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def compute_count_metrics(
    true_counts: Sequence[int],
    predicted_counts: Sequence[int],
) -> CountMetrics:
    """Compute MAE, RMSE, signed bias, means, and Pearson correlation."""
    if len(true_counts) != len(predicted_counts):
        raise ValueError("True and predicted counts must have the same length.")
    if not true_counts:
        raise ValueError("At least one count pair is required.")

    true_values = [float(value) for value in true_counts]
    predicted_values = [float(value) for value in predicted_counts]
    errors = [
        predicted - true
        for true, predicted in zip(true_values, predicted_values, strict=True)
    ]
    image_count = len(errors)
    mean_true = sum(true_values) / image_count
    mean_predicted = sum(predicted_values) / image_count

    true_variance_sum = sum((value - mean_true) ** 2 for value in true_values)
    predicted_variance_sum = sum(
        (value - mean_predicted) ** 2 for value in predicted_values
    )
    if true_variance_sum == 0 or predicted_variance_sum == 0:
        correlation = float("nan")
    else:
        covariance_sum = sum(
            (true - mean_true) * (predicted - mean_predicted)
            for true, predicted in zip(true_values, predicted_values, strict=True)
        )
        correlation = covariance_sum / sqrt(true_variance_sum * predicted_variance_sum)

    return CountMetrics(
        images=image_count,
        mae=sum(abs(error) for error in errors) / image_count,
        rmse=sqrt(sum(error**2 for error in errors) / image_count),
        bias=sum(errors) / image_count,
        mean_true=mean_true,
        mean_predicted=mean_predicted,
        correlation=correlation,
    )
