import numpy as np
import pytest


class DummyOutput:
    def __init__(self, name: str):
        self.name = name


class DummyGraph:
    def __init__(self, output_names):
        self.output = [DummyOutput(n) for n in output_names]


class DummyModel:
    def __init__(self, output_names):
        self.graph = DummyGraph(output_names)

    def SerializeToString(self) -> bytes:
        return b"dummy_model_bytes"


class DummyInput:
    def __init__(self, name: str, shape):
        self.name = name
        self.shape = shape


class DummySession:
    """
    Mimics onnxruntime.InferenceSession used by run_samples_for_bounds.
    """
    def __init__(self, model_bytes, providers=None, *, input_name="x", input_shape=None, outputs=None):
        self._input = DummyInput(input_name, input_shape if input_shape is not None else [None, 2])
        self._outputs = outputs  # list[np.ndarray] to return from run()

    def get_inputs(self):
        return [self._input]

    def run(self, output_names, feed):
        # basic sanity
        assert isinstance(output_names, list) and len(output_names) > 0
        assert isinstance(feed, dict) and self._input.name in feed
        if self._outputs is None:
            raise RuntimeError("DummySession: outputs not set")
        return self._outputs


class MeanMetric:
    def compute(self, output: np.ndarray) -> np.ndarray:
        # output shape: (num_samples, features)
        return np.mean(output, axis=1, keepdims=True)


def test_run_samples_for_bounds_happy_path(monkeypatch):
    """
    Happy-path:
    - valid num_samples
    - valid sampling_mode
    - metrics filtered by registry
    - model has inputs and outputs
    - outputs -> metric values -> returned dict
    """
    from nn_verification_visualisation.controller.process_manager import sample_runner

    # --- mock metric registry ---
    monkeypatch.setattr(sample_runner, "get_metric_map", lambda: {"mean": MeanMetric()}, raising=True)

    # --- mock ONNX load + modifier ---
    monkeypatch.setattr(sample_runner.onnx, "load", lambda path: DummyModel(["out0"]), raising=True)
    monkeypatch.setattr(sample_runner.NetworkModifier, "with_all_outputs", lambda m, sampling_mode: m, raising=True)

    # --- deterministic samples ---
    def fake_uniform(low, high, size):
        # size = (num_samples, len(bounds))
        return np.full(size, 0.5, dtype=np.float32)

    monkeypatch.setattr(sample_runner.np.random, "uniform", fake_uniform, raising=True)

    # --- mock ORT session ---
    # return one output tensor: (num_samples, 2)
    num_samples = 5
    outputs = [np.arange(num_samples * 2, dtype=np.float32).reshape(num_samples, 2)]

    def fake_inference_session(model_bytes, providers):
        return DummySession(model_bytes, providers=providers, input_name="x", input_shape=[None, 2], outputs=outputs)

    monkeypatch.setattr(sample_runner.ort, "InferenceSession", fake_inference_session, raising=True)

    # --- input ---
    network = type("N", (), {"path": "dummy.onnx"})()
    bounds = [(0.0, 1.0), (-2.0, 3.0)]

    result = sample_runner.run_samples_for_bounds(
        network=network,
        bounds=bounds,
        num_samples=num_samples,
        metrics=["mean", "unknown_metric_is_ignored"],
        sampling_mode=sample_runner.DEFAULT_SAMPLING_MODE,
    )

    assert result["num_samples"] == num_samples
    assert result["sampling_mode"] == sample_runner.DEFAULT_SAMPLING_MODE
    assert result["metrics"] == ["mean"]
    assert isinstance(result["outputs"], list) and len(result["outputs"]) == 1

    out0 = result["outputs"][0]
    assert out0["name"] == "out0"
    assert out0["shape"] == [2]  # output.shape[1:]
    assert "mean" in out0["values"]
    assert len(out0["values"]["mean"]) == num_samples


def test_run_samples_for_bounds_invalid_metrics_raises(monkeypatch):
    from nn_verification_visualisation.controller.process_manager import sample_runner

    monkeypatch.setattr(sample_runner, "get_metric_map", lambda: {"mean": MeanMetric()}, raising=True)
    network = type("N", (), {"path": "dummy.onnx"})()

    with pytest.raises(ValueError, match="No valid metrics selected"):
        sample_runner.run_samples_for_bounds(
            network=network,
            bounds=[(0.0, 1.0)],
            num_samples=10,
            metrics=["does_not_exist"],
        )


def test_run_samples_for_bounds_invalid_sampling_mode_raises(monkeypatch):
    from nn_verification_visualisation.controller.process_manager import sample_runner

    monkeypatch.setattr(sample_runner, "get_metric_map", lambda: {"mean": MeanMetric()}, raising=True)
    network = type("N", (), {"path": "dummy.onnx"})()

    with pytest.raises(ValueError, match="Invalid sampling mode"):
        sample_runner.run_samples_for_bounds(
            network=network,
            bounds=[(0.0, 1.0)],
            num_samples=10,
            metrics=["mean"],
            sampling_mode="nope",
        )


def test_run_samples_for_bounds_num_samples_limits(monkeypatch):
    from nn_verification_visualisation.controller.process_manager import sample_runner

    monkeypatch.setattr(sample_runner, "get_metric_map", lambda: {"mean": MeanMetric()}, raising=True)
    network = type("N", (), {"path": "dummy.onnx"})()

    with pytest.raises(ValueError, match="must be positive"):
        sample_runner.run_samples_for_bounds(network, [(0.0, 1.0)], 0, ["mean"])

    with pytest.raises(ValueError):
        sample_runner.run_samples_for_bounds(
            network, [(0.0, 1.0)], sample_runner.MAX_SAMPLES_PER_RUN + 1, ["mean"]
        )


def test_run_samples_for_bounds_dynamic_rank_shape_raises(monkeypatch):
    """
    Triggers the branch:
      if input_shape and len(input_shape) > 2:
        if not all(dim is not None for dim in expected_tail): raise
    """
    from nn_verification_visualisation.controller.process_manager import sample_runner

    monkeypatch.setattr(sample_runner, "get_metric_map", lambda: {"mean": MeanMetric()}, raising=True)
    monkeypatch.setattr(sample_runner.onnx, "load", lambda path: DummyModel(["out0"]), raising=True)
    monkeypatch.setattr(sample_runner.NetworkModifier, "with_all_outputs", lambda m, sampling_mode: m, raising=True)

    monkeypatch.setattr(
        sample_runner.np.random,
        "uniform",
        lambda low, high, size: np.full(size, 0.5, dtype=np.float32),
        raising=True,
    )

    # input_shape has dynamic tail dims -> should raise before session.run
    def fake_inference_session(model_bytes, providers):
        return DummySession(model_bytes, providers=providers, input_name="x", input_shape=[None, 2, None], outputs=None)

    monkeypatch.setattr(sample_runner.ort, "InferenceSession", fake_inference_session, raising=True)

    network = type("N", (), {"path": "dummy.onnx"})()

    with pytest.raises(RuntimeError, match="dynamic"):
        sample_runner.run_samples_for_bounds(
            network=network,
            bounds=[(0.0, 1.0), (0.0, 1.0)],  # len(bounds) irrelevant here
            num_samples=2,
            metrics=["mean"],
        )
