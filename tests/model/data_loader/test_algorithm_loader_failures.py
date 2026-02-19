from pathlib import Path


def test_algorithm_loader_missing_name_uses_default(tmp_path):
    """
    If ALGORITHM_NAME is missing, loader should still succeed and use a default name
    (usually derived from filename).
    """
    from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader

    f = tmp_path / "a.py"
    f.write_text(
        "IS_DETERMINISTIC = True\n"
        "def calculate_output_bounds(onnx_model, input_bounds):\n"
        "    return input_bounds\n",
        encoding="utf-8",
    )

    res = AlgorithmLoader().load_algorithm(str(f))
    assert res.is_success, res.error

    algo = res.data
    assert algo.path == str(f)
    assert isinstance(algo.name, str) and algo.name.strip() != ""
    # robust: default name usually contains filename stem
    assert f.stem in algo.name
    assert algo.is_deterministic is True


def test_algorithm_loader_missing_deterministic_uses_default(tmp_path):
    """
    If IS_DETERMINISTIC is missing, loader should still succeed and set a boolean default.
    """
    from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader

    f = tmp_path / "a.py"
    f.write_text(
        "ALGORITHM_NAME = 'A'\n"
        "def calculate_output_bounds(onnx_model, input_bounds):\n"
        "    return input_bounds\n",
        encoding="utf-8",
    )

    res = AlgorithmLoader().load_algorithm(str(f))
    assert res.is_success, res.error

    algo = res.data
    assert algo.path == str(f)
    assert algo.name == "A"
    assert isinstance(algo.is_deterministic, bool)


def test_algorithm_loader_syntax_error_fails(tmp_path):
    from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader

    f = tmp_path / "a.py"
    f.write_text(
        "ALGORITHM_NAME = 'A'\n"
        "IS_DETERMINISTIC = True\n"
        "def oops(:\n",
        encoding="utf-8",
    )

    res = AlgorithmLoader().load_algorithm(str(f))
    assert not res.is_success
