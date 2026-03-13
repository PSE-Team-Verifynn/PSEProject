import numpy as np
import pytest


class DummyMetric:
    def compute(self, output: np.ndarray):
        return np.mean(output, axis=1, keepdims=True)


class DummyModel:
    def __init__(self, names=("out0",)):
        class O:
            def __init__(self, name):
                self.name = name

        class G:
            def __init__(self, names):
                self.output = [O(n) for n in names]

        self.graph = G(names)

    def SerializeToString(self):
        return b"x"


class DummyInput:
    def __init__(self, name="x", shape=None):
        self.name = name
        self.shape = shape or [None, 2]


class DummySession:
    def __init__(self, outputs, shape=None, has_inputs=True):
        self.outputs = outputs
        self.shape = shape or [None, 2]
        self.has_inputs = has_inputs

    def get_inputs(self):
        return [] if not self.has_inputs else [DummyInput("x", self.shape)]

    def run(self, names, feed):
        assert "x" in feed
        return self.outputs


def _patch(monkeypatch, mod, *, outputs, model_outputs=("out0",), shape=None, has_inputs=True):
    monkeypatch.setattr(mod, "get_metric_map", lambda: {"mean": DummyMetric()}, raising=True)
    monkeypatch.setattr(mod.onnx, "load", lambda path: DummyModel(model_outputs), raising=True)
    monkeypatch.setattr(mod.NetworkModifier, "with_all_outputs", lambda m, sampling_mode: m, raising=True)
    monkeypatch.setattr(
        mod.np.random,
        "uniform",
        lambda low, high, size: np.full(size, 0.5, dtype=np.float32),
        raising=True,
    )
    monkeypatch.setattr(
        mod.ort,
        "InferenceSession",
        lambda *_a, **_k: DummySession(outputs=outputs, shape=shape, has_inputs=has_inputs),
        raising=True,
    )


def test_run_samples_success(monkeypatch):
    from nn_verification_visualisation.controller.process_manager import sample_runner as mod

    captured = {}

    def fake_uniform(low, high, size):
        arr = np.arange(np.prod(size), dtype=np.float32).reshape(size)
        return arr

    class CapturingSession(DummySession):
        def run(self, names, feed):
            assert "x" in feed
            captured["x_shape"] = feed["x"].shape
            return self.outputs

    monkeypatch.setattr(mod, "get_metric_map", lambda: {"mean": DummyMetric()}, raising=True)
    monkeypatch.setattr(mod.onnx, "load", lambda path: DummyModel(("out0",)), raising=True)
    monkeypatch.setattr(mod.NetworkModifier, "with_all_outputs", lambda m, sampling_mode: m, raising=True)
    monkeypatch.setattr(mod.np.random, "uniform", fake_uniform, raising=True)
    monkeypatch.setattr(
        mod.ort,
        "InferenceSession",
        lambda *_a, **_k: CapturingSession(
            outputs=[np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            shape=[None, 1, 2],
        ),
        raising=True,
    )

    net = type("N", (), {"path": "dummy.onnx"})()
    res = mod.run_samples_for_bounds(
        network=net,
        bounds=[(0.0, 1.0), (-2.0, 3.0)],
        num_samples=3,
        metrics=["mean"],
        sampling_mode=mod.DEFAULT_SAMPLING_MODE,
    )

    assert captured["x_shape"] == (3, 1, 2)
    assert res["num_samples"] == 3
    assert res["metrics"] == ["mean"]
    assert len(res["outputs"]) == 1
    assert res["outputs"][0]["name"] == "out0"
    assert res["outputs"][0]["shape"] == [1]
    assert res["outputs"][0]["values"]["mean"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "num_samples,metrics,sampling_mode",
    [
        (0, ["mean"], None),
        (-1, ["mean"], None),
        (10_000_000, ["mean"], None),
        (1, ["unknown"], None),
        (1, ["mean"], "bad_mode"),
    ],
)
def test_run_samples_validation_errors(monkeypatch, num_samples, metrics, sampling_mode):
    from nn_verification_visualisation.controller.process_manager import sample_runner as mod

    monkeypatch.setattr(mod, "get_metric_map", lambda: {"mean": DummyMetric()}, raising=True)
    net = type("N", (), {"path": "dummy.onnx"})()

    with pytest.raises(ValueError):
        mod.run_samples_for_bounds(
            network=net,
            bounds=[(0.0, 1.0)],
            num_samples=num_samples,
            metrics=metrics,
            sampling_mode=sampling_mode or mod.DEFAULT_SAMPLING_MODE,
        )


def test_run_samples_no_outputs_or_inputs(monkeypatch):
    from nn_verification_visualisation.controller.process_manager import sample_runner as mod

    net = type("N", (), {"path": "dummy.onnx"})()

    # no outputs declared in model
    _patch(monkeypatch, mod, outputs=[], model_outputs=())
    with pytest.raises(RuntimeError, match="Model has no outputs to sample"):
        mod.run_samples_for_bounds(net, [(0.0, 1.0)], 2, ["mean"])

    # no outputs produced in single-batch branch
    _patch(monkeypatch, mod, outputs=[], model_outputs=("out0",), shape=[1, 1])
    with pytest.raises(RuntimeError, match="Model produced no outputs"):
        mod.run_samples_for_bounds(net, [(0.0, 1.0)], 2, ["mean"])

    # no outputs produced in normal batch branch
    _patch(monkeypatch, mod, outputs=[], model_outputs=("out0",), shape=[None, 1])
    with pytest.raises(RuntimeError, match="Model produced no outputs"):
        mod.run_samples_for_bounds(net, [(0.0, 1.0)], 2, ["mean"])

    # no inputs
    _patch(monkeypatch, mod, outputs=[np.zeros((2, 1), dtype=np.float32)], has_inputs=False)
    with pytest.raises(RuntimeError, match="Model has no inputs"):
        mod.run_samples_for_bounds(net, [(0.0, 1.0)], 2, ["mean"])


def test_run_samples_dynamic_shape_error(monkeypatch):
    from nn_verification_visualisation.controller.process_manager import sample_runner as mod

    net = type("N", (), {"path": "dummy.onnx"})()

    _patch(
        monkeypatch,
        mod,
        outputs=[np.zeros((2, 1), dtype=np.float32)],
        shape=[None, 2, None],
    )

    with pytest.raises(RuntimeError, match="Sample input rank mismatch"):
        mod.run_samples_for_bounds(
            network=net,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            num_samples=2,
            metrics=["mean"],
        )

    _patch(
        monkeypatch,
        mod,
        outputs=[np.zeros((2, 1), dtype=np.float32)],
        shape=[None, 2, 2],
    )

    with pytest.raises(RuntimeError, match="Sample input size mismatch"):
        mod.run_samples_for_bounds(
            network=net,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            num_samples=2,
            metrics=["mean"],
        )