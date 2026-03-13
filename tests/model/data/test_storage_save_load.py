import onnx
from onnx import helper, TensorProto
from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig

def _make_dummy_onnx(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, producer_name="pytest")
    onnx.save(model, str(path))


def test_storage_save_and_load(tmp_path, qapp):
    from nn_verification_visualisation.model.data.storage import Storage
    from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
    from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
    from nn_verification_visualisation.model.data.input_bounds import InputBounds
    from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
    from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader

    model_path = tmp_path / "net.onnx"
    _make_dummy_onnx(model_path)

    cfg = NetworkVerificationConfig(
        network=NeuralNetwork("N", str(model_path), onnx.load(str(model_path))),
        layers_dimensions=[1, 1],
    )
    cfg.bounds = InputBounds(1)
    if hasattr(cfg.bounds, "load_list"):
        cfg.bounds.load_list([(0.0, 1.0)])

    algo = Algorithm(name="A", path="dummy_algo.py", is_deterministic=True)
    pgc = PlotGenerationConfig(
        nnconfig=cfg,
        algorithm=algo,
        selected_neurons=[(0, 0)],
        parameters=[],
        bounds_index=-1,
    )

    d = DiagramConfig(
        plot_generation_configs=[pgc],
        polygons=[[(0.0, 0.0), (1.0, 0.0)]],
    )
    d.plots = [[0]]

    s = Storage()
    save_path = tmp_path / "save_state.json"
    s.set_save_state_path(str(save_path))

    s.networks = [cfg]
    s.diagrams = [d]

    res = s.save_to_disk()
    assert res.is_success, res.error
    assert save_path.exists()

    s.networks = []
    s.diagrams = []

    res2 = s.load_from_disk()
    assert res2.is_success, res2.error

    # IMPORTANT: loader can succeed but skip networks if ONNX path is invalid
    state = res2.data
    warnings = SaveStateLoader().get_warnings()

    assert len(state.loaded_networks) == 1, f"Loaded 0 networks. Warnings: {warnings}"
    assert len(state.diagrams) == 1

    # if your Storage doesn't auto-apply, apply explicitly (works in both designs)
    if hasattr(s, "load_save_state"):
        s.load_save_state(state)

    assert len(s.networks) == 1
    assert len(s.diagrams) == 1

def test_storage_request_autosave_without_qt_app_calls_save_immediately(monkeypatch):
    import nn_verification_visualisation.model.data.storage as st_mod
    from nn_verification_visualisation.model.data.storage import Storage

    # Force "no Qt app"
    monkeypatch.setattr(st_mod.QCoreApplication, "instance", lambda: None, raising=True)

    calls = {"n": 0}

    def fake_save(self):
        calls["n"] += 1
        # return value not important here

    monkeypatch.setattr(Storage, "save_to_disk", fake_save, raising=True)

    s = Storage()
    s._suppress_autosave = False
    s.request_autosave()

    assert calls["n"] == 1

    def test_storage_request_autosave_creates_timer_and_starts(monkeypatch):
        import nn_verification_visualisation.model.data.storage as st_mod
        from nn_verification_visualisation.model.data.storage import Storage

        # Force "Qt app exists"
        monkeypatch.setattr(st_mod.QCoreApplication, "instance", lambda: object(), raising=True)

        # Spy timer implementation
        calls = {"created": 0, "single": 0, "connected": 0, "started": 0, "delay": None}

        class DummySignal:
            def connect(self, fn):
                calls["connected"] += 1

        class DummyTimer:
            def __init__(self):
                calls["created"] += 1
                self.timeout = DummySignal()

            def setSingleShot(self, v: bool):
                calls["single"] += 1

            def start(self, delay_ms: int):
                calls["started"] += 1
                calls["delay"] = delay_ms

        monkeypatch.setattr(st_mod, "QTimer", DummyTimer, raising=True)

        s = Storage()

        # suppress branch
        s._suppress_autosave = True
        s.request_autosave()
        assert calls["created"] == 0
        assert calls["started"] == 0

        # real branch: create timer and start it
        s._suppress_autosave = False
        s._autosave_timer = None
        s._autosave_delay_ms = 123

        s.request_autosave()

        assert calls["created"] == 1
        assert calls["single"] == 1
        assert calls["connected"] == 1
        assert calls["started"] == 1
        assert calls["delay"] == 123

        # second call: must reuse timer, only start again
        s.request_autosave()
        assert calls["created"] == 1
        assert calls["started"] == 2

def test_storage_save_to_disk_exporter_failure_and_algorithm_listeners(monkeypatch, tmp_path):
    import nn_verification_visualisation.model.data.storage as st_mod
    from nn_verification_visualisation.model.data.storage import Storage
    from nn_verification_visualisation.utils.result import Failure

    # Prepare storage with temp save path
    s = Storage()
    save_path = tmp_path / "save_state.json"
    s.set_save_state_path(str(save_path))

    # Force exporter failure
    class DummyExporter:
        def export_save_state(self, _state):
            return Failure(RuntimeError("export failed"))

    monkeypatch.setattr(st_mod, "SaveStateExporter", lambda: DummyExporter(), raising=True)

    res = s.save_to_disk()
    assert not res.is_success
    assert isinstance(res.error, RuntimeError)
    assert "export failed" in str(res.error)

    # ---- algorithms list + listeners coverage ----
    calls = {"n": 0}
    s.algorithm_change_listeners = [lambda: calls.__setitem__("n", calls["n"] + 1)]

    class Algo:
        def __init__(self, path: str):
            self.path = path

    a1 = Algo("a.py")
    a2 = Algo("b.py")

    # add -> listener
    s.algorithms = []
    s.add_algorithm(a1)
    assert len(s.algorithms) == 1
    assert calls["n"] == 1

    # modify non-existing -> no listener
    s.modify_algorithm("nope.py", a2)
    assert calls["n"] == 1

    # modify existing -> listener
    s.modify_algorithm("a.py", a2)
    assert s.algorithms[0] is a2
    assert calls["n"] == 2

    # remove non-existing -> no listener
    s.remove_algorithm("nope.py")
    assert calls["n"] == 2

    # remove existing -> listener
    s.remove_algorithm("b.py")
    assert s.algorithms == []
    assert calls["n"] == 3