import onnx
from onnx import helper, TensorProto


def _make_dummy_onnx(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, producer_name="pytest")
    onnx.save(model, str(path))


def test_neural_network_loader_loads_onnx(tmp_path):
    from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader

    model_path = tmp_path / "net.onnx"
    _make_dummy_onnx(model_path)

    res = NeuralNetworkLoader().load_neural_network(str(model_path))
    assert res.is_success, res.error
    nn = res.data
    assert nn.path == str(model_path)
    assert nn.model is not None
