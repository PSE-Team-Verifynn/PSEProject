import pytest
from unittest.mock import Mock
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QWidget

from nn_verification_visualisation.view.base_view.main_window import MainWindow
from tests.conftest import mock_color_manager

_MODULE = "nn_verification_visualisation.view"


@pytest.fixture
def mocked_main_window(mocker, qtbot, mock_color_manager):
    """Set up MainWindow with mocked BaseView and InfoPopup."""
    mock_active_view = Mock()
    # Must be a real QWidget so setCentralWidget() accepts it
    mock_base_view = QWidget()
    mock_base_view.active_view = mock_active_view

    PATCH_PATH = "nn_verification_visualisation.view.base_view.main_window"

    mocker.patch(f"{PATCH_PATH}.BaseView", return_value=mock_base_view)
    mock_info_popup_class = mocker.patch(f"{PATCH_PATH}.InfoPopup")

    window = MainWindow(mock_color_manager)
    qtbot.addWidget(window)

    yield {
        "window": window,
        "base_view": mock_base_view,
        "active_view": mock_active_view,
        "InfoPopup": mock_info_popup_class,
    }

    # Ensure teardown works: ignore the custom close logic
    window.exit_confirmed = True
    window.close()


class TestMainWindowInit:
    def test_window_title(self, mocked_main_window):
        assert mocked_main_window["window"].windowTitle() == "PSE Neuron App"

    def test_initial_size(self, mocked_main_window):
        window = mocked_main_window["window"]
        assert window.width() == 800
        assert window.height() == 600

    def test_exit_confirmed_initially_false(self, mocked_main_window):
        assert mocked_main_window["window"].exit_confirmed is False

    def test_exit_dialog_open_initially_false(self, mocked_main_window):
        assert mocked_main_window["window"].exit_dialog_open is False

    def test_base_view_set_as_central_widget(self, mocked_main_window):
        window = mocked_main_window["window"]
        assert window.centralWidget() == mocked_main_window["base_view"]


class TestCloseEventAlreadyConfirmed:
    def test_accepts_event_when_exit_confirmed(self, mocked_main_window):
        window = mocked_main_window["window"]
        window.exit_confirmed = True

        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        event.accept.assert_called_once()
        event.ignore.assert_not_called()

    def test_does_not_open_dialog_when_exit_confirmed(self, mocked_main_window):
        window = mocked_main_window["window"]
        window.exit_confirmed = True

        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        mocked_main_window["InfoPopup"].assert_not_called()


class TestCloseEventFirstAttempt:
    def test_ignores_event_on_first_close(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        event.ignore.assert_called_once()
        event.accept.assert_not_called()

    def test_sets_exit_dialog_open(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        assert window.exit_dialog_open is True

    def test_opens_dialog_on_active_view(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        mocked_main_window["active_view"].open_dialog.assert_called_once()

    def test_info_popup_created_with_warning_type(self, mocked_main_window):
        from nn_verification_visualisation.view.dialogs.info_type import InfoType

        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        call_args = mocked_main_window["InfoPopup"].call_args
        assert call_args[0][2] == InfoType.WARNING

    def test_info_popup_receives_two_buttons(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        buttons = mocked_main_window["InfoPopup"].call_args[0][3]
        assert len(buttons) == 2

    def test_dialog_buttons_have_correct_object_names(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        buttons = mocked_main_window["InfoPopup"].call_args[0][3]
        object_names = {b.objectName() for b in buttons}
        assert "light-button" in object_names
        assert "error-button" in object_names


class TestCloseEventDialogAlreadyOpen:
    def test_does_not_open_second_dialog_when_dialog_already_open(self, mocked_main_window):
        window = mocked_main_window["window"]
        window.exit_dialog_open = True

        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        mocked_main_window["InfoPopup"].assert_not_called()

    def test_still_ignores_event_when_dialog_already_open(self, mocked_main_window):
        window = mocked_main_window["window"]
        window.exit_dialog_open = True

        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        event.ignore.assert_called_once()


class TestCancelButton:
    def test_cancel_closes_dialog_on_active_view(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        cancel_button = next(
            b for b in mocked_main_window["InfoPopup"].call_args[0][3]
            if b.objectName() == "light-button"
        )
        cancel_button.click()

        mocked_main_window["active_view"].close_dialog.assert_called_once()

    def test_cancel_resets_exit_dialog_open(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        cancel_button = next(
            b for b in mocked_main_window["InfoPopup"].call_args[0][3]
            if b.objectName() == "light-button"
        )
        cancel_button.click()

        assert window.exit_dialog_open is False

    def test_cancel_does_not_set_exit_confirmed(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        cancel_button = next(
            b for b in mocked_main_window["InfoPopup"].call_args[0][3]
            if b.objectName() == "light-button"
        )
        cancel_button.click()

        assert window.exit_confirmed is False


class TestConfirmButton:
    def test_confirm_sets_exit_confirmed(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        confirm_button = next(
            b for b in mocked_main_window["InfoPopup"].call_args[0][3]
            if b.objectName() == "error-button"
        )
        confirm_button.click()

        assert window.exit_confirmed is True

    def test_confirm_closes_dialog_on_active_view(self, mocked_main_window):
        window = mocked_main_window["window"]
        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        confirm_button = next(
            b for b in mocked_main_window["InfoPopup"].call_args[0][3]
            if b.objectName() == "error-button"
        )
        confirm_button.click()

        mocked_main_window["active_view"].close_dialog.assert_called_once()

    def test_confirm_triggers_window_close(self, mocked_main_window, mocker):
        window = mocked_main_window["window"]
        mock_close = mocker.patch.object(window, "close")

        event = Mock(spec=QCloseEvent)
        window.closeEvent(event)

        confirm_button = next(
            b for b in mocked_main_window["InfoPopup"].call_args[0][3]
            if b.objectName() == "error-button"
        )
        confirm_button.click()

        mock_close.assert_called_once()