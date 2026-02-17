"""Tests for main application entry point."""
import sys

import pytest


@pytest.fixture
def mocked_main(mocker):
    """Set up common mocks for main() tests."""
    mock_app = mocker.Mock()
    mock_app.exec.return_value = 0

    mock_color_manager = mocker.Mock()
    mock_window = mocker.Mock()
    mock_storage = mocker.Mock()

    mocks = {
        'qapp_class': mocker.patch('nn_verification_visualisation.__main__.QApplication', return_value=mock_app),
        'color_manager_class': mocker.patch('nn_verification_visualisation.__main__.ColorManager', return_value=mock_color_manager),
        'main_window_class': mocker.patch('nn_verification_visualisation.__main__.MainWindow', return_value=mock_window),
        'storage_class': mocker.patch('nn_verification_visualisation.__main__.Storage', return_value=mock_storage),
        'mp': mocker.patch('nn_verification_visualisation.__main__.mp'),
        'sys_exit': mocker.patch('sys.exit'),
        'app': mock_app,
        'color_manager': mock_color_manager,
        'window': mock_window,
        'storage': mock_storage,
    }
    return mocks

class TestMain:
    """Tests for the main() function."""

    def test_main_creates_application(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['qapp_class'].assert_called_once_with(sys.argv)

    def test_main_sets_fusion_style(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['app'].setStyle.assert_called_once_with("Fusion")

    def test_main_initializes_color_manager(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['color_manager_class'].assert_called_once_with(mocked_main['app'])
        mocked_main['color_manager'].load_raw.assert_called_once_with(
            ":src/nn_verification_visualisation/style.qss"
        )

    def test_main_storage_initialization(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['storage_class'].assert_called_once()

    def test_main_storage_save_state_path_set(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['storage'].set_save_state_path.assert_called_once()
        actual_path = mocked_main['storage'].set_save_state_path.call_args[0][0]
        assert actual_path.endswith("save_state.json")

    def test_main_storage_save_state_tried_to_load_from_disk(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['storage'].load_from_disk.assert_called_once()

    def test_main_activates_freeze_support(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['mp'].freeze_support.assert_called_once()

    def test_main_creates_and_shows_window(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['main_window_class'].assert_called_once_with(mocked_main['color_manager'])
        mocked_main['window'].showMaximized.assert_called_once()

    def test_main_connect_to_about_to_quit(self, mocked_main):
        from nn_verification_visualisation.__main__ import main
        main()
        mocked_main['app'].aboutToQuit.connect.assert_called_once()