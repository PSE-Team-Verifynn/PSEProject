import os
import pytest
from PySide6.QtWidgets import QApplication
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def reset_storage_singleton(request):
    # If a test uses mock_storage, Storage is patched -> don't touch it
    if "mock_storage" in request.fixturenames:
        yield
        return

    try:
        from nn_verification_visualisation.model.data.storage import Storage
        s = Storage()
        if hasattr(s, "networks"):
            s.networks = []
        if hasattr(s, "diagrams"):
            s.diagrams = []
        if hasattr(s, "algorithms"):
            s.algorithms = []
        if hasattr(s, "algorithm_change_listeners"):
            s.algorithm_change_listeners = []
    except Exception:
        pass

    yield


@pytest.fixture
def mock_storage():
    with patch("nn_verification_visualisation.model.data.storage.Storage") as mock:
        instance = Mock()
        instance.networks = []
        instance.diagrams = []
        instance.algorithms = []
        instance.algorithm_change_listeners = []
        instance.request_autosave = Mock()
        instance.save_to_disk = Mock()
        instance.load_from_disk = Mock()
        mock.return_value = instance
        yield instance
