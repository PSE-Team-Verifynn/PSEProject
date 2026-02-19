from pathlib import Path


class DummyNetworkConfig:
    def __init__(self, n: int):
        self.layers_dimensions = [n]


def test_csv_wrong_column_count_fails(tmp_path: Path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    p = tmp_path / "b.csv"
    p.write_text("a,b,c,d\n1,2,3,4\n", encoding="utf-8")

    res = InputBoundsLoader().load_input_bounds(str(p), DummyNetworkConfig(1))
    assert not res.is_success


def test_vnnlib_invalid_expression_fails(tmp_path: Path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    p = tmp_path / "p.vnnlib"
    p.write_text("(assert (and (>= X_0 0) (<= X_0 1) )) ))", encoding="utf-8")

    res = InputBoundsLoader().load_input_bounds(str(p), DummyNetworkConfig(1))
    assert not res.is_success
