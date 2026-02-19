def test_algorithm_loader_loads_simple_algorithm(tmp_path):
    from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader

    algo_file = tmp_path / "my_algo.py"
    algo_file.write_text(
        "\n".join(
            [
                "ALGORITHM_NAME = 'My Test Algo'",
                "IS_DETERMINISTIC = True",
                "",
                "import numpy as np",
                "",
                "def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:",
                "    return np.asarray(input_bounds, dtype=float)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    res = AlgorithmLoader().load_algorithm(str(algo_file))
    assert res.is_success, res.error
    algo = res.data
    assert algo.name == "My Test Algo"
    assert algo.path == str(algo_file)
    assert algo.is_deterministic is True
