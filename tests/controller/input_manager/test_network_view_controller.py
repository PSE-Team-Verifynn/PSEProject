from PySide6.QtCore import Qt
from onnx import helper, TensorProto


def _model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w1 = helper.make_tensor("W1", TensorProto.FLOAT, [4, 3], [0.0] * 12)
    w2 = helper.make_tensor("W2", TensorProto.FLOAT, [3, 2], [0.0] * 6)
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y], initializer=[w1, w2])
    return helper.make_model(g, producer_name="pytest")


class View:
    def __init__(self, path=None):
        self.path = path
        self.dialogs = []
        self.tabs = []

    def open_network_file_picker(self, _): return self.path
    def open_dialog(self, d): self.dialogs.append(d)
    def close_dialog(self): return True
    def add_network_tab(self, c): self.tabs.append(c)
    def close_network_tab(self, _): pass


def _cfg(InputBounds):
    class Net: name = "N"
    class Cfg:
        network = Net()
        layers_dimensions = [2, 1]
        bounds = InputBounds(2)
        saved_bounds = []
        selected_bounds_index = -1
    return Cfg()


def test_load_bounds_variants(monkeypatch, qapp):
    import nn_verification_visualisation.controller.input_manager.network_view_controller as mod
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader
    from nn_verification_visualisation.utils.result import Success, Failure

    monkeypatch.setattr(mod, "InfoType", type("T", (), {"CONFIRMATION": 1, "ERROR": 2}), raising=True)
    monkeypatch.setattr(mod, "InfoPopup", lambda *a, **k: object(), raising=True)

    # cancel
    ctrl = mod.NetworkViewController(View(None))
    assert ctrl.load_bounds(type("C", (), {"layers_dimensions": [2]})()) is False

    # success
    monkeypatch.setattr(InputBoundsLoader, "load_input_bounds",
                        lambda self, p, c: Success({0: (0.0, 1.0), 1: (2.0, 3.0)}), raising=True)
    ctrl = mod.NetworkViewController(View("x.csv"))
    hit = {"ok": False}
    monkeypatch.setattr(ctrl, "_apply_loaded_bounds", lambda c, d: hit.__setitem__("ok", d == {0: (0.0, 1.0), 1: (2.0, 3.0)}), raising=True)
    assert ctrl.load_bounds(type("C", (), {"layers_dimensions": [2]})()) is True
    assert hit["ok"] is True

    # failure
    monkeypatch.setattr(InputBoundsLoader, "load_input_bounds",
                        lambda self, p, c: Failure(RuntimeError("bad")), raising=True)
    ctrl = mod.NetworkViewController(View("x.csv"))
    assert ctrl.load_bounds(type("C", (), {"layers_dimensions": [2]})()) is False


def test_load_new_network_and_remove(monkeypatch, qapp):
    import nn_verification_visualisation.controller.input_manager.network_view_controller as mod
    from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader
    from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
    from nn_verification_visualisation.model.data.storage import Storage
    from nn_verification_visualisation.utils.result import Success, Failure

    monkeypatch.setattr(mod, "InfoType", type("T", (), {"ERROR": 1}), raising=True)
    monkeypatch.setattr(mod, "InfoPopup", lambda *a, **k: object(), raising=True)
    monkeypatch.setattr(Storage, "request_autosave", lambda self: None, raising=True)

    # success
    nn = NeuralNetwork("N", "a.onnx", _model())
    monkeypatch.setattr(NeuralNetworkLoader, "load_neural_network", lambda self, p: Success(nn), raising=True)
    v = View("a.onnx")
    ctrl = mod.NetworkViewController(v)
    Storage().networks = []

    cfg = ctrl.load_new_network()
    assert cfg is not None
    assert cfg.layers_dimensions == [4, 3, 2]
    assert len(Storage().networks) == 1
    assert v.tabs == [cfg]

    # remove existing + missing
    assert ctrl.remove_neural_network(cfg) is True
    assert ctrl.remove_neural_network(cfg) is False

    # failure
    monkeypatch.setattr(NeuralNetworkLoader, "load_neural_network",
                        lambda self, p: Failure(RuntimeError("bad")), raising=True)
    assert mod.NetworkViewController(View("b.onnx")).load_new_network() is None


def test_bounds_workflow_and_signal(monkeypatch, qapp):
    import nn_verification_visualisation.controller.input_manager.network_view_controller as mod
    from nn_verification_visualisation.model.data.input_bounds import InputBounds
    from nn_verification_visualisation.model.data.storage import Storage

    hits = {"n": 0}
    monkeypatch.setattr(Storage, "request_autosave", lambda self: hits.__setitem__("n", hits["n"] + 1), raising=True)

    ctrl = mod.NetworkViewController(View())
    cfg = _cfg(InputBounds)

    # save twice
    cfg.bounds.load_list([(0.0, 1.0), (2.0, 3.0)])
    assert ctrl.save_bounds(cfg) == 0
    cfg.bounds.load_list([(-1.0, 0.0), (10.0, 11.0)])
    assert ctrl.save_bounds(cfg) == 1
    assert len(cfg.saved_bounds) == 2

    # select saved, then draft
    ctrl.select_bounds(cfg, 0)
    assert cfg.bounds.get_values() == [(0.0, 1.0), (2.0, 3.0)]
    ctrl.select_bounds(cfg, None)
    assert cfg.bounds.get_values() == [(0.0, 0.0), (0.0, 0.0)]

    # apply loaded bounds to current and then to selected saved
    ctrl._apply_loaded_bounds(cfg, {0: (5.0, 6.0), 1: (7.0, 8.0)})
    assert cfg.bounds.get_values() == [(5.0, 6.0), (7.0, 8.0)]
    cfg.selected_bounds_index = 0
    ctrl._apply_loaded_bounds(cfg, {0: (1.0, 2.0), 1: (3.0, 4.0)})
    assert cfg.saved_bounds[0].get_values() == [(1.0, 2.0), (3.0, 4.0)]

    # remove bounds
    assert ctrl.remove_bounds(cfg, 1) is True
    assert ctrl.remove_bounds(cfg, 0) is True
    assert cfg.saved_bounds == []

    # signal hookup
    ctrl._connect_bounds_autosave(cfg)
    cfg.bounds.setData(cfg.bounds.index(0, 0), 9.0, role=Qt.ItemDataRole.EditRole)
    assert hits["n"] >= 1