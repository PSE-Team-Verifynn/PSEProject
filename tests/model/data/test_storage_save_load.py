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
