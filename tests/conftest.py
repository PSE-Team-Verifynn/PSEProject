"""Shared test fixtures for all tests."""
import pytest
from PySide6.QtWidgets import QApplication
from unittest.mock import Mock, patch

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