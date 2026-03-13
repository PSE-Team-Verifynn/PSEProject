"""Shared test fixtures for all tests."""
import pytest
from PySide6.QtWidgets import QApplication
from unittest.mock import Mock, patch

import os
from pathlib import Path

from nn_verification_visualisation.view.base_view.color_manager import ColorManager


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication once per test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

@pytest.fixture
def mock_storage():
    """Mock the Storage singleton."""
    with patch('nn_verification_visualisation.model.data.storage.Storage') as mock:
        instance = Mock()
        instance.networks = []
        instance.algorithms = []
        instance.algorithm_change_listeners = []
        mock.return_value = instance
        yield instance

@pytest.fixture
def sample_network_config():
    """Create a sample network config for testing."""
    config = Mock()
    config.network.name = "TestNetwork"
    config.network.path = "/test/path"
    config.layers_dimensions = [10, 20, 15, 5]
    config.saved_bounds = []
    config.selected_bounds_index = -1
    return config

@pytest.fixture
def mock_color_manager():
    """Mocks the ColorManager."""
    return Mock(spec=ColorManager)

def pytest_addoption(parser):
    parser.addoption(
        "--vnncomp-dir",
        default=os.environ.get("VNNCOMP_DIR", ""),
        help="Root directory of the VNN-COMP benchmark suite. If not set, integration tests are skipped.",
    )


def _collect_vnncomp_pairs(root: Path, sample: int = 1, seed: int = 42, category_filter: str = ""):
    import random
    rng = random.Random(seed)  # isolated RNG — never touches global random state

    for csv_file in sorted(root.rglob("instances.csv")):
        category = csv_file.parent.name
        if category_filter and category != category_filter:
            continue

        all_pairs = []
        for line in csv_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            onnx_path   = csv_file.parent / parts[0]
            vnnlib_path = csv_file.parent / parts[1]
            timeout     = float(parts[2]) if len(parts) >= 3 else 60.0
            if onnx_path.exists() and vnnlib_path.exists():
                all_pairs.append((onnx_path, vnnlib_path, category, timeout))

        if not all_pairs:
            continue

        # sample=0 means "take all"
        chosen = rng.sample(all_pairs, min(sample, len(all_pairs))) if sample > 0 else all_pairs
        yield from chosen