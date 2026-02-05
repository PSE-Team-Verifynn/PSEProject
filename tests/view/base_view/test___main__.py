"""Tests for main application entry point."""
import sys


class TestMain:
    """Tests for the main() function."""

    def test_main_creates_application(self, mocker):
        """Test that main() creates a QApplication instance."""
        from nn_verification_visualisation.__main__ import main

        # Mock using pytest-mock's mocker fixture
        mock_qapp_class = mocker.patch('nn_verification_visualisation.__main__.QApplication')
        mock_color_manager_class = mocker.patch('nn_verification_visualisation.__main__.ColorManager')
        mock_main_window_class = mocker.patch('nn_verification_visualisation.__main__.MainWindow')
        mock_exit = mocker.patch('sys.exit')

        # Setup
        mock_app = mocker.Mock()
        mock_qapp_class.return_value = mock_app
        mock_app.exec.return_value = 0

        # Run
        main()

        # Verify
        mock_qapp_class.assert_called_once_with(sys.argv)

    def test_main_sets_fusion_style(self, mocker):
        """Test that main() sets the Fusion style."""
        from nn_verification_visualisation.__main__ import main

        mock_qapp_class = mocker.patch('nn_verification_visualisation.__main__.QApplication')
        mocker.patch('nn_verification_visualisation.__main__.ColorManager')
        mocker.patch('nn_verification_visualisation.__main__.MainWindow')
        mocker.patch('sys.exit')

        mock_app = mocker.Mock()
        mock_qapp_class.return_value = mock_app
        mock_app.exec.return_value = 0

        main()

        mock_app.setStyle.assert_called_once_with("Fusion")

    def test_main_initializes_color_manager(self, mocker):
        """Test that main() creates and initializes ColorManager."""
        from nn_verification_visualisation.__main__ import main

        mock_qapp_class = mocker.patch('nn_verification_visualisation.__main__.QApplication')
        mock_color_manager_class = mocker.patch('nn_verification_visualisation.__main__.ColorManager')
        mocker.patch('nn_verification_visualisation.__main__.MainWindow')
        mocker.patch('sys.exit')

        mock_app = mocker.Mock()
        mock_qapp_class.return_value = mock_app
        mock_app.exec.return_value = 0

        mock_color_manager = mocker.Mock()
        mock_color_manager_class.return_value = mock_color_manager

        main()

        mock_color_manager_class.assert_called_once_with(mock_app)
        mock_color_manager.load_raw.assert_called_once_with(
            ":src/nn_verification_visualisation/style.qss"
        )

    def test_main_creates_and_shows_window(self, mocker):
        """Test that main() creates MainWindow and shows it."""
        from nn_verification_visualisation.__main__ import main

        mock_qapp_class = mocker.patch('nn_verification_visualisation.__main__.QApplication')
        mock_color_manager_class = mocker.patch('nn_verification_visualisation.__main__.ColorManager')
        mock_main_window_class = mocker.patch('nn_verification_visualisation.__main__.MainWindow')
        mocker.patch('sys.exit')

        mock_app = mocker.Mock()
        mock_qapp_class.return_value = mock_app
        mock_app.exec.return_value = 0

        mock_color_manager = mocker.Mock()
        mock_color_manager_class.return_value = mock_color_manager

        mock_window = mocker.Mock()
        mock_main_window_class.return_value = mock_window

        main()

        mock_main_window_class.assert_called_once_with(mock_color_manager)
        mock_window.show.assert_called_once()