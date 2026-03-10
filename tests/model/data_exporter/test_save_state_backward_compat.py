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


def test_save_state_loader_accepts_old_bounds_format(tmp_path, qapp):
    from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader

    model_path = tmp_path / "net.onnx"
    _make_dummy_onnx(model_path)

    # emulate old exporter: bounds as list-of-pairs, saved_bounds list-of-lists
    doc = {
        "format": "nnvv_save_state",
        "version": 1,
        "loaded_networks": [
            {
                "network": {"name": "N", "path": str(model_path)},
                "layers_dimensions": [1, 1],
                "activation_values": [],
                "selected_bounds_index": -1,
                "bounds": [[0.0, 1.0]],
                "saved_bounds": [[[2.0, 3.0]]],
            }
        ],
        "diagrams": [],
    }

    f = tmp_path / "save_state.json"
    f.write_text(json.dumps(doc), encoding="utf-8")

    res = SaveStateLoader().load_save_state(str(f))
    assert res.is_success, res.error
    assert len(res.data.loaded_networks) == 1
