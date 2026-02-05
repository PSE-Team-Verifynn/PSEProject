from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SampleMetric:
    key: str
    name: str
    compute: Callable[[np.ndarray], np.ndarray]


def load_metrics() -> list[SampleMetric]:
    return [
        SampleMetric(
            key="max",
            name="Max activation",
            compute=lambda output: np.max(output, axis=0),
        ),
        SampleMetric(
            key="mean",
            name="Mean activation",
            compute=lambda output: np.mean(output, axis=0),
        ),
    ]


def get_metric_map() -> dict[str, SampleMetric]:
    return {metric.key: metric for metric in load_metrics()}
