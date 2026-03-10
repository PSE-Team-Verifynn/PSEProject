def test_neural_network_loader_missing_file_fails(tmp_path):
    from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader

    p = tmp_path / "nope.onnx"
    res = NeuralNetworkLoader().load_neural_network(str(p))
    assert not res.is_success


def test_neural_network_loader_corrupt_file_fails(tmp_path):
    from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader

    p = tmp_path / "bad.onnx"
    p.write_bytes(b"not an onnx file")

    res = NeuralNetworkLoader().load_neural_network(str(p))
    assert not res.is_success
