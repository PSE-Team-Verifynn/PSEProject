import numpy as np

from nn_verification_visualisation.controller.process_manager.sample_metric_registry import (
    SampleMetric,
    get_metric_map,
    load_metrics,
)


def test_load_metrics_returns_expected_metric_definitions():
    metrics = load_metrics()

    assert len(metrics) == 3
    assert all(isinstance(metric, SampleMetric) for metric in metrics)
    assert [metric.key for metric in metrics] == ["max", "mean", "range"]
    assert [metric.name for metric in metrics] == [
        "Max Activation",
        "Mean Activation",
        "Activation Range",
    ]


def test_metric_compute_functions_produce_expected_values():
    output = np.array(
        [
            [1.0, -2.0, 3.0],
            [-4.0, 5.0, -6.0],
            [7.0, -8.0, 9.0],
        ],
        dtype=np.float32,
    )

    metric_map = get_metric_map()

    np.testing.assert_allclose(metric_map["max"].compute(output), np.array([7.0, 8.0, 9.0]))
    np.testing.assert_allclose(metric_map["mean"].compute(output), np.array([4.0 / 3.0, -5.0 / 3.0, 2.0]))
    np.testing.assert_allclose(metric_map["range"].compute(output), np.array([11.0, 13.0, 15.0]))


def test_get_metric_map_is_keyed_by_metric_key():
    metric_map = get_metric_map()

    assert set(metric_map.keys()) == {"max", "mean", "range"}
    assert metric_map["max"].name == "Max Activation"
    assert metric_map["mean"].name == "Mean Activation"
    assert metric_map["range"].name == "Activation Range"
