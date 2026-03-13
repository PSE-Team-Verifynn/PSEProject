import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from PySide6.QtWidgets import QWidget, QMainWindow

from nn_verification_visualisation.view.base_view.action_menu import ActionMenu
from nn_verification_visualisation.view.dialogs.info_type import InfoType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_parent():
    parent = Mock(spec=QWidget)
    parent.open_dialog = Mock()
    parent.close_dialog = Mock()
    parent.parent = Mock(return_value=None)
    parent.__class__.__name__ = "InsertView"
    return parent


@pytest.fixture
def action_menu(qapp, mock_parent):
    return ActionMenu(mock_parent)


@pytest.fixture
def mock_loader(mocker):
    """Patches SaveStateLoader; returns the mock instance."""
    loader = Mock()
    mocker.patch(
        "nn_verification_visualisation.view.base_view.action_menu.SaveStateLoader",
        return_value=loader,
    )
    return loader


@pytest.fixture
def mock_exporter(mocker):
    """Patches SaveStateExporter and Storage; returns the exporter mock instance."""
    exporter = Mock()
    mocker.patch(
        "nn_verification_visualisation.view.base_view.action_menu.SaveStateExporter",
        return_value=exporter,
    )
    mocker.patch(
        "nn_verification_visualisation.view.base_view.action_menu.Storage",
        return_value=Mock(),
    )
    return exporter


@pytest.fixture
def mock_info_popup(mocker):
    """Patches InfoPopup; returns the class mock so call args can be inspected."""
    return mocker.patch("nn_verification_visualisation.view.base_view.action_menu.InfoPopup")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_file_returning(mocker, path):
    mocker.patch(
        "nn_verification_visualisation.view.base_view.action_menu.QFileDialog.getOpenFileName",
        return_value=(path, ""),
    )


def _save_file_returning(mocker, path):
    mocker.patch(
        "nn_verification_visualisation.view.base_view.action_menu.QFileDialog.getSaveFileName",
        return_value=(path, ""),
    )


def _last_popup_type(mock_info_popup):
    return mock_info_popup.call_args[0][2]


def _last_popup_msg(mock_info_popup):
    return mock_info_popup.call_args[0][1]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestActionMenuInit:
    def test_menu_has_correct_actions(self, action_menu):
        titles = [a.text() for a in action_menu.menu.actions()]
        assert titles == ["Settings", "Open Project", "Export Project", "Exit"]

    def test_parent_stored_and_shadow_applied(self, action_menu, mock_parent):
        assert action_menu.parent is mock_parent
        assert action_menu.menu.graphicsEffect() is not None


# ---------------------------------------------------------------------------
# Settings action
# ---------------------------------------------------------------------------

class TestSettingsAction:
    def test_opens_settings_dialog_with_close_callback(self, action_menu, mock_parent, mocker):
        mock_cls = mocker.patch(
            "nn_verification_visualisation.view.base_view.action_menu.SettingsDialog"
        )
        action_menu.menu.actions()[0].trigger()

        mock_cls.assert_called_once_with(mock_parent.close_dialog)
        mock_parent.open_dialog.assert_called_once_with(mock_cls.return_value)


# ---------------------------------------------------------------------------
# Open Project action
# ---------------------------------------------------------------------------

class TestOpenProjectAction:
    def test_cancelled_dialog_does_nothing(self, action_menu, mock_parent, mocker):
        _open_file_returning(mocker, "")
        action_menu.menu.actions()[1].trigger()
        mock_parent.open_dialog.assert_not_called()

    def test_load_failure_shows_warning(self, action_menu, mocker, mock_loader, mock_info_popup):
        _open_file_returning(mocker, "/some/file.json")
        mock_loader.load_save_state.return_value = Mock(is_success=False, error="Bad file")

        action_menu.menu.actions()[1].trigger()

        assert _last_popup_type(mock_info_popup) == InfoType.WARNING
        assert "Could not open project" in _last_popup_msg(mock_info_popup)

    def test_successful_load_updates_storage_and_shows_confirmation(
        self, action_menu, mocker, mock_loader, mock_info_popup
    ):
        _open_file_returning(mocker, "/some/file.json")
        save_state = Mock()
        mock_loader.load_save_state.return_value = Mock(is_success=True, data=save_state)
        mock_loader.get_warnings.return_value = []
        mock_storage = Mock()
        mocker.patch(
            "nn_verification_visualisation.view.base_view.action_menu.Storage",
            return_value=mock_storage,
        )

        action_menu.menu.actions()[1].trigger()

        mock_storage.load_save_state.assert_called_once_with(save_state)
        mock_storage.save_to_disk.assert_called_once()
        assert _last_popup_type(mock_info_popup) == InfoType.CONFIRMATION

    def test_load_with_warnings_shows_warning_popup(
        self, action_menu, mocker, mock_loader, mock_info_popup
    ):
        _open_file_returning(mocker, "/some/file.json")
        mock_loader.load_save_state.return_value = Mock(is_success=True, data=Mock())
        mock_loader.get_warnings.return_value = ["Warning A"]
        mocker.patch(
            "nn_verification_visualisation.view.base_view.action_menu.Storage",
            return_value=Mock(),
        )

        action_menu.menu.actions()[1].trigger()

        assert _last_popup_type(mock_info_popup) == InfoType.WARNING
        assert "Warning A" in _last_popup_msg(mock_info_popup)

    def test_successful_load_calls_reload_on_base_view(
        self, action_menu, mocker, mock_loader, mock_info_popup
    ):
        _open_file_returning(mocker, "/some/file.json")
        mock_loader.load_save_state.return_value = Mock(is_success=True, data=Mock())
        mock_loader.get_warnings.return_value = []
        mocker.patch(
            "nn_verification_visualisation.view.base_view.action_menu.Storage",
            return_value=Mock(),
        )
        mock_base_view = Mock()
        mocker.patch.object(
            action_menu, "_ActionMenu__find_base_view", return_value=mock_base_view
        )

        action_menu.menu.actions()[1].trigger()

        mock_base_view.reload_from_storage.assert_called_once()


# ---------------------------------------------------------------------------
# Export Project action
# ---------------------------------------------------------------------------

class TestExportProjectAction:
    def test_cancelled_dialog_does_nothing(self, action_menu, mock_parent, mocker):
        _save_file_returning(mocker, "")
        action_menu.menu.actions()[2].trigger()
        mock_parent.open_dialog.assert_not_called()

    def test_exporter_failure_shows_error(self, action_menu, mocker, mock_exporter, mock_info_popup):
        _save_file_returning(mocker, "/out/project.json")
        mock_exporter.export_save_state.return_value = Mock(is_success=False, error="Err")

        action_menu.menu.actions()[2].trigger()

        assert _last_popup_type(mock_info_popup) == InfoType.ERROR
        assert "Could not export project" in _last_popup_msg(mock_info_popup)

    def test_write_failure_shows_error(self, action_menu, mocker, mock_exporter, mock_info_popup, tmp_path):
        _save_file_returning(mocker, str(tmp_path / "out.json"))
        mock_exporter.export_save_state.return_value = Mock(is_success=True, data="{}")
        mocker.patch(
            "nn_verification_visualisation.view.base_view.action_menu.Path.write_text",
            side_effect=OSError("disk full"),
        )

        action_menu.menu.actions()[2].trigger()

        assert _last_popup_type(mock_info_popup) == InfoType.ERROR

    def test_successful_export_shows_confirmation(self, action_menu, mocker, mock_exporter, mock_info_popup, tmp_path):
        _save_file_returning(mocker, str(tmp_path / "out.json"))
        mock_exporter.export_save_state.return_value = Mock(is_success=True, data="{}")

        action_menu.menu.actions()[2].trigger()

        assert _last_popup_type(mock_info_popup) == InfoType.CONFIRMATION

    def test_missing_json_suffix_is_appended(self, action_menu, mocker, mock_exporter, mock_info_popup, tmp_path):
        _save_file_returning(mocker, str(tmp_path / "myexport"))   # no .json suffix
        mock_exporter.export_save_state.return_value = Mock(is_success=True, data="{}")

        action_menu.menu.actions()[2].trigger()

        assert _last_popup_msg(mock_info_popup).endswith(".json")
        assert ".json.json" not in _last_popup_msg(mock_info_popup)


# ---------------------------------------------------------------------------
# __find_base_view
# ---------------------------------------------------------------------------

class TestFindBaseView:
    def test_returns_none_when_not_found(self, action_menu, mock_parent):
        assert action_menu._ActionMenu__find_base_view() is None

    def test_returns_immediate_parent_when_it_is_base_view(self, action_menu, mock_parent):
        mock_parent.__class__.__name__ = "BaseView"
        assert action_menu._ActionMenu__find_base_view() is mock_parent

    def test_finds_base_view_further_up_hierarchy(self, action_menu, mock_parent):
        grandparent = Mock()
        grandparent.__class__.__name__ = "BaseView"
        grandparent.parent.return_value = None
        mock_parent.__class__.__name__ = "Other"
        mock_parent.parent.return_value = grandparent

        assert action_menu._ActionMenu__find_base_view() is grandparent


# ---------------------------------------------------------------------------
# Exit action
# ---------------------------------------------------------------------------

class TestExitAction:
    def test_saves_storage_and_closes_main_window(self, action_menu, mocker, qapp, qtbot):
        mock_storage = Mock()
        mocker.patch(
            "nn_verification_visualisation.view.base_view.action_menu.Storage",
            return_value=mock_storage,
        )
        main_window = QMainWindow()
        qtbot.addWidget(main_window)

        with patch(
            "nn_verification_visualisation.view.base_view.action_menu.QApplication.instance",
            return_value=qapp,
        ), patch.object(qapp, "topLevelWidgets", return_value=[main_window]):
            action_menu.menu.actions()[3].trigger()

        mock_storage.save_to_disk.assert_called_once()

    def test_no_main_window_does_not_crash(self, action_menu, mocker, qapp):
        mocker.patch(
            "nn_verification_visualisation.view.base_view.action_menu.Storage",
            return_value=Mock(),
        )
        with patch(
            "nn_verification_visualisation.view.base_view.action_menu.QApplication.instance",
            return_value=qapp,
        ), patch.object(qapp, "topLevelWidgets", return_value=[]):
            action_menu.menu.actions()[3].trigger()  # must not raise