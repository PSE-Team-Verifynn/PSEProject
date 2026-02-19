from pathlib import Path


class DummyNetworkConfig:
    """InputBoundsLoader uses only layers_dimensions[0]."""
    def __init__(self, input_count: int):
        self.layers_dimensions = [input_count]


def test_csv_two_columns_ok(tmp_path: Path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    p = tmp_path / "bounds.csv"
    p.write_text("lo,hi\n0,1\n-2,3.5\n", encoding="utf-8")

    res = InputBoundsLoader().load_input_bounds(str(p), DummyNetworkConfig(2))
    assert res.is_success, res.error
    assert res.data == {0: (0.0, 1.0), 1: (-2.0, 3.5)}


def test_vnnlib_and_ok(tmp_path: Path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    p = tmp_path / "prop.vnnlib"
    p.write_text(
        """
(assert (and
    (>= X_0 -1)
    (<= X_0  1)
    (>= X_1  0)
    (<= X_1  2)
))
(assert (<= Y_0 1)) ; must be ignored
""",
        encoding="utf-8",
    )

    res = InputBoundsLoader().load_input_bounds(str(p), DummyNetworkConfig(2))
    assert res.is_success, res.error
    assert res.data == {0: (-1.0, 1.0), 1: (0.0, 2.0)}


def test_vnnlib_or_bounding_box_ok(tmp_path: Path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    p = tmp_path / "prop.vnnlib"
    p.write_text(
        """
(assert (or
    (and (>= X_0 0) (<= X_0 1) (>= X_1 0) (<= X_1 1))
    (and (>= X_0 2) (<= X_0 3) (>= X_1 -1) (<= X_1 0))
))
""",
        encoding="utf-8",
    )

    res = InputBoundsLoader().load_input_bounds(str(p), DummyNetworkConfig(2))
    assert res.is_success, res.error
    assert res.data == {0: (0.0, 3.0), 1: (-1.0, 1.0)}
