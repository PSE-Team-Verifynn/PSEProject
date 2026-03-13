import json

import onnx
from onnx import helper, TensorProto


def _make_dummy_onnx(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, producer_name="pytest")
    onnx.save(model, str(path))


def test_save_state_loader_restores_samples(tmp_path, qapp):
    from nn_verification_visualisation.model.data_exporter.save_state_exporter import SaveStateExporter
    from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader

    from nn_verification_visualisation.model.data.save_state import SaveState
    from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
    from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
    from nn_verification_visualisation.model.data.input_bounds import InputBounds
    from nn_verification_visualisation.model.data.diagram_config import DiagramConfig

    model_path = tmp_path / "net.onnx"
    _make_dummy_onnx(model_path)

    nn = NeuralNetwork("N", str(model_path), onnx.load(str(model_path)))
    cfg = NetworkVerificationConfig(network=nn, layers_dimensions=[1, 1])

    b = InputBounds(1)
    if hasattr(b, "load_list"):
        b.load_list([(0.0, 1.0)])
    cfg.bounds = b

    sb = InputBounds(1)
    if hasattr(sb, "load_list"):
        sb.load_list([(2.0, 3.0)])
    sample = {"m": 1.0, "arr": [1, 2, 3]}
    if hasattr(sb, "set_sample"):
        sb.set_sample(sample)
    cfg.saved_bounds = [sb]

    d = DiagramConfig(plot_generation_configs=[], polygons=[])
    d.plots = []

    state = SaveState(loaded_networks=[cfg], diagrams=[d])

    exp = SaveStateExporter().export_save_state(state)
    assert exp.is_success, exp.error

    f = tmp_path / "save_state.json"
    f.write_text(exp.data, encoding="utf-8")

    loaded = SaveStateLoader().load_save_state(str(f))
    assert loaded.is_success, loaded.error

    st = loaded.data
    assert len(st.loaded_networks) == 1
    lb = st.loaded_networks[0].saved_bounds[0]
    if hasattr(lb, "get_sample"):
        assert lb.get_sample() == sample

def _make_dummy_onnx(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, producer_name="pytest")
    onnx.save(model, str(path))


def test_save_state_loader_rejects_wrong_format(tmp_path, qapp):
    from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader

    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"format": "WRONG"}), encoding="utf-8")

    res = SaveStateLoader().load_save_state(str(f))
    assert not res.is_success


def test_save_state_loader_collects_warning_and_skips_missing_onnx(monkeypatch, tmp_path, qapp):
    """
    If onnx.load fails, loader should add a warning and skip that network (but still succeed).
    """
    from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader

    doc = {
        "format": "nnvv_save_state",
        "version": 1,
        "loaded_networks": [
            {
                "network": {"name": "N", "path": str(tmp_path / "missing.onnx")},
                "layers_dimensions": [1, 1],
                "activation_values": [],
                "selected_bounds_index": -1,
                "bounds": {"values": [[0.0, 1.0]], "sample": None},
                "saved_bounds": [],
            }
        ],
        "diagrams": [],
    }

    f = tmp_path / "state.json"
    f.write_text(json.dumps(doc), encoding="utf-8")

    # force onnx.load to throw
    import nn_verification_visualisation.model.data_loader.save_state_loader as mod
    monkeypatch.setattr(mod.onnx, "load", lambda p: (_ for _ in ()).throw(FileNotFoundError(p)), raising=True)

    loader = SaveStateLoader()
    res = loader.load_save_state(str(f))
    assert res.is_success, res.error

    assert len(res.data.loaded_networks) == 0
    warnings = loader.get_warnings()
    assert any("missing" in w.lower() or "not found" in w.lower() for w in warnings)


def test_save_state_loader_filters_invalid_diagrams(tmp_path, qapp):
    """
    Diagram with no valid plot_generation_configs should be skipped (current behavior).
    """
    from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader

    doc = {
        "format": "nnvv_save_state",
        "version": 1,
        "loaded_networks": [],
        "diagrams": [
            {"plot_generation_configs": [], "polygons": [], "plots": []}
        ],
    }

    f = tmp_path / "state.json"
    f.write_text(json.dumps(doc), encoding="utf-8")

    res = SaveStateLoader().load_save_state(str(f))
    assert res.is_success, res.error
    assert res.data.diagrams == []