from pathlib import Path

import numpy as np
import pytest

from nn_verification_visualisation.controller.input_manager.network_view_controller import NetworkViewController
from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader
from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.controller.process_manager.sample_runner import run_samples_for_bounds
from tests.conftest import _collect_vnncomp_pairs

_VNNCOMP_SAMPLE    = 1   # instances per category
_VNNCOMP_SEED      = 42  # reproducible selection

def pytest_generate_tests(metafunc):
    if "vnncomp_case" not in metafunc.fixturenames:
        return
    raw = metafunc.config.getoption("--vnncomp-dir", default="")
    if not raw:
        metafunc.parametrize("vnncomp_case", [])
        return

    pairs = list(_collect_vnncomp_pairs(Path(raw), sample=_VNNCOMP_SAMPLE, seed=_VNNCOMP_SEED))
    ids = [f"{cat}/{Path(o).stem}+{Path(v).stem}" for o, v, cat, _ in pairs]
    metafunc.parametrize("vnncomp_case", pairs, ids=ids)

def _resolve_algorithm_dir() -> Path:
    """Resolves to <project_root>/algorithms."""
    return Path(__file__).parents[1] / "algorithms"


@pytest.fixture
def loaded_case(vnncomp_case, qapp):
    onnx_path, vnnlib_path, category, timeout = vnncomp_case

    nn_result = NeuralNetworkLoader().load_neural_network(str(onnx_path))
    if not nn_result.is_success:
        pytest.skip(f"Network load failed: {nn_result.error}")
    nn = nn_result.data

    # Consistent layer dim extraction
    layer_dims = NetworkViewController.get_layer_dimensions_from_network(nn)
    config = NetworkVerificationConfig(nn, layer_dims)

    bounds_result = InputBoundsLoader().load_input_bounds(str(vnnlib_path), config)
    if not bounds_result.is_success:
        pytest.skip(f"Bounds load failed: {bounds_result.error}")

    return nn, bounds_result.data, category, timeout


class TestVnncompIntegration:

    def test_network_loads(self, vnncomp_case):          # No qapp needed here
        onnx_path, _, _, _ = vnncomp_case
        result = NeuralNetworkLoader().load_neural_network(str(onnx_path))
        assert result.is_success, f"Failed: {result.error}"

    def test_bounds_load(self, loaded_case):
        _, bounds_dict, _, _ = loaded_case
        assert bounds_dict, "Bounds dict is empty"


    def test_box_on_first_two_nodes(self, loaded_case, request):
        """Run box_ibp_numpy on neurons (0,0) and (0,1) of every sampled network."""
        nn, bounds_dict, category, timeout = loaded_case

        # Resolve algorithm path
        algo_dir = _resolve_algorithm_dir()
        algo_path = algo_dir / "input_bounds_check.py"
        if not algo_path.exists():
            pytest.skip(f"box_no_autolirpa.py not found at {algo_path}")

        layer_dims = NetworkViewController.get_layer_dimensions_from_network(nn)

        # Guard: need at least 2 neurons in layer 0 to test both nodes
        layer_0_size = layer_dims[0] if layer_dims else 0
        if layer_0_size < 1:
            pytest.skip(f"Layer 0 has no neurons in {category}")

        selected_neurons = [(0, 0)]
        if layer_0_size >= 2:
            selected_neurons.append((0, 1))
        else:
            pytest.skip(f"Layer 0 has only 1 neuron in {category}")

        # Convert bounds_dict -> np.ndarray shape (N, 2)
        n = len(bounds_dict)
        input_bounds = np.array(
            [[bounds_dict[i][0], bounds_dict[i][1]] for i in range(n)],
            dtype=float,
        )

        # Execute
        result = AlgorithmExecutor().execute_algorithm(
            model=nn.model,
            input_bounds=input_bounds,
            algorithm_path=str(algo_path),
            selected_neurons=selected_neurons,
            num_directions=2,
        )

        assert result.is_success, (
            f"[{category}] simple_zonotope failed on neurons {selected_neurons}: {result.error}"
        )

        output_bounds, directions = result.data

        # Shape check: one bound pair per direction
        assert output_bounds.shape[1] == 2, (
            f"Expected output shape (D, 2), got {output_bounds.shape}"
        )
        assert len(output_bounds) == len(directions), (
            f"Output bounds rows ({len(output_bounds)}) != directions ({len(directions)})"
        )

        # Sanity: every output bound must be a valid interval
        for i, (lo, hi) in enumerate(output_bounds.tolist()):
            assert lo <= hi, (
                f"[{category}] Output bound {i} is invalid: {lo} > {hi}"
            )