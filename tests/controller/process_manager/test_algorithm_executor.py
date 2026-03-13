import numpy as np
from onnx import helper, TensorProto


def _model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    return helper.make_model(g, producer_name="pytest")


class Result:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error
        self.is_success = error is None


def test_input_bounds_to_numpy_paths_and_error():
    from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor

    class B1:
        pass

    b1 = B1()
    setattr(b1, "_InputBounds__value", [(0.0, 1.0), (-2.0, 3.0)])
    assert np.allclose(
        AlgorithmExecutor.input_bounds_to_numpy(b1),
        np.array([[0.0, 1.0], [-2.0, 3.0]], dtype=float),
    )

    class B2:
        def rowCount(self): return 2
        def index(self, r, c): return (r, c)
        def data(self, idx): return {(0, 0): "0", (0, 1): "1.5", (1, 0): -2, (1, 1): 3}[idx]

    assert np.allclose(
        AlgorithmExecutor.input_bounds_to_numpy(B2()),
        np.array([[0.0, 1.5], [-2.0, 3.0]], dtype=float),
    )

    class B3:
        def rowCount(self): return 1
        def index(self, r, c): return (r, c)
        def data(self, idx): return 0.0 if idx == (0, 0) else None

    try:
        AlgorithmExecutor.input_bounds_to_numpy(B3())
        assert False
    except ValueError:
        pass


def test_execute_algorithm_success_and_failures(monkeypatch):
    import nn_verification_visualisation.controller.process_manager.algorithm_executor as mod

    model = _model()
    bounds = np.array([[0.0, 1.0]], dtype=float)

    ex = mod.AlgorithmExecutor()

    # success
    monkeypatch.setattr(mod.NetworkModifier, "custom_output_layer", lambda self, *a, **k: "M", raising=True)
    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_calculate_output_bounds",
        staticmethod(lambda p: Result(data=lambda m, b: np.array([[10.0, 11.0]], dtype=float))),
        raising=True,
    )
    r = ex.execute_algorithm(model, bounds, "a.py", [(0, 0)], 2)
    assert r.is_success
    assert np.allclose(r.data[0], np.array([[10.0, 11.0]], dtype=float))
    assert len(r.data[1]) == 2

    # loader failure
    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_calculate_output_bounds",
        staticmethod(lambda p: Result(error=RuntimeError("bad algo"))),
        raising=True,
    )
    r = ex.execute_algorithm(model, bounds, "a.py", [(0, 0)], 2)
    assert not r.is_success and isinstance(r.error, RuntimeError)

    # modifier failure
    monkeypatch.setattr(
        mod.NetworkModifier,
        "custom_output_layer",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("bad modifier")),
        raising=True,
    )
    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_calculate_output_bounds",
        staticmethod(lambda p: Result(data=lambda m, b: b)),
        raising=True,
    )
    r = ex.execute_algorithm(model, bounds, "a.py", [(0, 0)], 2)
    assert not r.is_success and isinstance(r.error, RuntimeError)

    # algorithm failure
    monkeypatch.setattr(mod.NetworkModifier, "custom_output_layer", lambda self, *a, **k: "M", raising=True)
    monkeypatch.setattr(
        mod.AlgorithmLoader,
        "load_calculate_output_bounds",
        staticmethod(lambda p: Result(data=lambda m, b: (_ for _ in ()).throw(ValueError("fail")))),
        raising=True,
    )
    r = ex.execute_algorithm(model, bounds, "a.py", [(0, 0)], 2)
    assert not r.is_success and isinstance(r.error, ValueError)