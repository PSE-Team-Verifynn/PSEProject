from pathlib import Path
import pytest


class DummyNetworkConfig:
    """InputBoundsLoader uses only layers_dimensions[0]."""
    def __init__(self, n: int):
        self.layers_dimensions = [n]


def run(loader, tmp_path: Path, name: str, content: str, n_inputs: int):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return loader.load_input_bounds(str(p), DummyNetworkConfig(n_inputs))


# -------------------------
# load_input_bounds() guards
# -------------------------
def test_load_input_bounds_rejects_unsupported_extension(tmp_path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    p = tmp_path / "bounds.txt"
    p.write_text("x", encoding="utf-8")

    res = InputBoundsLoader().load_input_bounds(str(p), DummyNetworkConfig(1))
    assert not res.is_success


def test_load_input_bounds_rejects_invalid_network_config(tmp_path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    p = tmp_path / "bounds.csv"
    p.write_text("lo,hi\n0,1\n", encoding="utf-8")

    class BadCfg:
        layers_dimensions = []  # invalid

    res = InputBoundsLoader().load_input_bounds(str(p), BadCfg())
    assert not res.is_success


# -------------------------
# CSV success paths
# -------------------------
def test_csv_two_columns_success(tmp_path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    res = run(
        InputBoundsLoader(),
        tmp_path,
        "b.csv",
        "lo,hi\n0,1\n-2,3.5\n",
        n_inputs=2,
    )
    assert res.is_success, res.error
    assert res.data == {0: (0.0, 1.0), 1: (-2.0, 3.5)}


def test_csv_three_columns_permutation_success(tmp_path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    res = run(
        InputBoundsLoader(),
        tmp_path,
        "b.csv",
        "idx,lo,hi\n1,-2,3\n0,0,1\n",
        n_inputs=2,
    )
    assert res.is_success, res.error
    assert res.data == {0: (0.0, 1.0), 1: (-2.0, 3.0)}


# -------------------------
# CSV error branches (many lines covered via parametrization)
# -------------------------
@pytest.mark.parametrize(
    "name,content,n_inputs",
    [
        # empty file -> StopIteration branch
        ("empty.csv", "", 1),
        # wrong field count (not 2 or 3)
        ("badcols.csv", "a,b,c,d\n1,2,3,4\n", 1),
        # inconsistent row length
        ("inconsistent.csv", "lo,hi\n0,1\n0,1,2\n", 2),
        # row count mismatch
        ("rowcount.csv", "lo,hi\n0,1\n", 2),
        # 3 columns: index not int
        ("idx_not_int.csv", "idx,lo,hi\nx,0,1\n0,0,1\n", 2),
        # 3 columns: indices not a permutation of 0..N-1
        ("idx_bad_perm.csv", "idx,lo,hi\n0,0,1\n2,0,1\n", 2),
        # float parse error
        ("nan.csv", "lo,hi\nx,1\n0,1\n", 2),
        # lower > upper
        ("order.csv", "lo,hi\n2,1\n0,1\n", 2),
    ],
)
def test_csv_error_cases(tmp_path, name, content, n_inputs):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    res = run(InputBoundsLoader(), tmp_path, name, content, n_inputs=n_inputs)
    assert not res.is_success


def test_csv_oserror_branch(tmp_path):
    """Triggers OSError in __parse_csv(open(...))."""
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    missing = tmp_path / "missing.csv"
    res = InputBoundsLoader().load_input_bounds(str(missing), DummyNetworkConfig(1))
    assert not res.is_success


# -------------------------
# VNNLIB success paths (covers atomic variations + Y ignored + OR)
# -------------------------
def test_vnnlib_and_success_with_swapped_comparison_and_equal(tmp_path):
    """
    Covers atomic():
      - (<= 0.5 X_0) meaning X_0 >= 0.5
      - (= X_1 1.0)
    Also ensures Y constraints are ignored.
    """
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    text = """
; comment
(assert (and
    (<= 0.5 X_0)   ; X_0 >= 0.5
    (<= X_0  1.5)  ; X_0 <= 1.5
    (=  X_1  1.0)
))
(assert (<= Y_0 1)) ; ignored
"""
    res = run(InputBoundsLoader(), tmp_path, "p.vnnlib", text, n_inputs=2)
    assert res.is_success, res.error
    assert res.data == {0: (0.5, 1.5), 1: (1.0, 1.0)}


def test_vnnlib_or_bounding_box_success(tmp_path):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    text = """
(assert (or
  (and (>= X_0 0) (<= X_0 1) (>= X_1 0) (<= X_1 1))
  (and (>= X_0 2) (<= X_0 3) (>= X_1 -1) (<= X_1 0))
))
"""
    res = run(InputBoundsLoader(), tmp_path, "p.vnnlib", text, n_inputs=2)
    assert res.is_success, res.error
    assert res.data == {0: (0.0, 3.0), 1: (-1.0, 1.0)}


# -------------------------
# VNNLIB error branches
# -------------------------
@pytest.mark.parametrize(
    "name,text,n_inputs",
    [
        # invalid expression: extra closing paren -> stack empty on ')'
        ("bad_paren1.vnnlib", "(assert (and (>= X_0 0) (<= X_0 1))) ))", 1),
        # invalid expression: missing closing paren -> stack not empty after parse
        ("bad_paren2.vnnlib", "(assert (and (>= X_0 0) (<= X_0 1))", 1),
        # no X specs at all -> "No input specs for X_i"
        ("no_x.vnnlib", "(assert (<= Y_0 1))", 1),
        # missing one bound -> "Missing bounds for X_1"
        ("missing_bound.vnnlib", "(assert (and (>= X_0 0) (<= X_0 1) (>= X_1 0)))", 2),
        # contradictory bounds -> merge returns None -> "Invalid input constraints"
        ("contradiction.vnnlib", "(assert (and (>= X_0 2) (<= X_0 1)))", 1),
    ],
)
def test_vnnlib_error_cases(tmp_path, name, text, n_inputs):
    from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader

    res = run(InputBoundsLoader(), tmp_path, name, text, n_inputs=n_inputs)
    assert not res.is_success