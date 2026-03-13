from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QWidget, QPushButton

from nn_verification_visualisation.model.data.input_bounds import InputBounds
from nn_verification_visualisation.view.network_view.network_page import NetworkPage
from nn_verification_visualisation.view.network_view.network_view import NetworkView


def _make_bounds(values, sample=None):
    obj = MagicMock()
    obj.get_values.return_value = values
    obj.get_sample.return_value = sample
    return obj


def _make_config():
    bounds = InputBounds(2)
    bounds.load_list([(0.0, 1.0), (2.0, 3.0)])
    return SimpleNamespace(
        network=SimpleNamespace(name="Net-1"),
        layers_dimensions=[2, 1],
        bounds=bounds,
        saved_bounds=[],
        selected_bounds_index=-1,
    )


def _mock_insert_view_dependencies():
    tabs = QWidget()
    tabs.close_tab = Mock()
    tabs.setCornerWidget = Mock()
    tabs.reset = Mock()
    tabs.add_tab = Mock()
    tabs.tabBar = Mock(return_value=Mock(height=Mock(return_value=40), sizeHint=Mock(return_value=QSize(100, 40))))

    action_menu = Mock()
    action_menu.menu = Mock()
    return tabs, action_menu


def test_network_page_full_results_opens_dialog_for_selected_bounds(qapp, monkeypatch):
    config = _make_config()
    result = {"num_samples": 2, "metrics": [], "outputs": []}
    config.saved_bounds = [_make_bounds([(0.0, 1.0), (2.0, 3.0)], sample=result)]
    config.selected_bounds_index = 0

    current_network_view = MagicMock()
    current_network_view.close_dialog = Mock()
    controller = MagicMock(current_network_view=current_network_view)

    monkeypatch.setattr(
        "nn_verification_visualisation.view.network_view.network_page.NetworkWidget",
        lambda *_args, **_kwargs: QWidget(),
    )

    page = NetworkPage(controller, config)

    page._NetworkPage__on_full_results_clicked()

    current_network_view.open_dialog.assert_called_once()
    dialog = current_network_view.open_dialog.call_args.args[0]
    assert dialog.result == result


def test_network_page_add_bounds_enables_edit_mode(qapp, monkeypatch):
    config = _make_config()
    controller = MagicMock()
    monkeypatch.setattr(
        "nn_verification_visualisation.view.network_view.network_page.NetworkWidget",
        lambda *_args, **_kwargs: QWidget(),
    )

    page = NetworkPage(controller, config)
    page.show()

    page._NetworkPage__on_add_bounds_clicked()

    assert page.edit_group.isVisible() is True
    assert page.add_button.isVisible() is False


def test_network_view_initializes_from_storage_and_opens_confirmation_dialog(qapp):
    change_view = Mock()
    storage = MagicMock()
    storage.networks = [SimpleNamespace(network=SimpleNamespace(name="Stored Net"))]
    tabs, action_menu = _mock_insert_view_dependencies()

    with (
        patch("nn_verification_visualisation.view.network_view.network_view.NetworkViewController") as MockController,
        patch("nn_verification_visualisation.view.network_view.network_view.Storage", return_value=storage),
        patch("nn_verification_visualisation.view.base_view.insert_view.Tabs", return_value=tabs),
        patch("nn_verification_visualisation.view.base_view.insert_view.ActionMenu", return_value=action_menu),
        patch.object(NetworkView, "add_network_tab") as add_network_tab,
        patch("nn_verification_visualisation.view.network_view.network_view.InfoPopup") as MockInfoPopup,
    ):
        controller = MockController.return_value
        view = NetworkView(change_view)

        add_network_tab.assert_called_once_with(storage.networks[0])

        popup = Mock()
        MockInfoPopup.return_value = popup
        view.open_dialog = Mock()
        view.close_dialog = Mock()
        view.close_tab(0)

        view.open_dialog.assert_called_once_with(popup)
        buttons = MockInfoPopup.call_args.args[3]
        confirm_button = next(button for button in buttons if isinstance(button, QPushButton) and button.text() == "Continue")
        confirm_button.click()

    controller.remove_neural_network.assert_called_once_with(storage.networks[0])


def test_network_view_reload_resets_tabs_and_rebuilds_from_storage(qapp):
    storage = MagicMock()
    storage.networks = [SimpleNamespace(network=SimpleNamespace(name="A")), SimpleNamespace(network=SimpleNamespace(name="B"))]
    tabs, action_menu = _mock_insert_view_dependencies()

    with (
        patch("nn_verification_visualisation.view.network_view.network_view.NetworkViewController"),
        patch("nn_verification_visualisation.view.network_view.network_view.Storage", return_value=storage),
        patch("nn_verification_visualisation.view.base_view.insert_view.Tabs", return_value=tabs),
        patch("nn_verification_visualisation.view.base_view.insert_view.ActionMenu", return_value=action_menu),
        patch.object(NetworkView, "add_network_tab") as add_network_tab,
    ):
        view = NetworkView(Mock())
        view.controller = MagicMock()

        view.reload_from_storage()

    tabs.reset.assert_called_once()
    view.controller.connect_all_bounds_autosave.assert_called_once()
    assert add_network_tab.call_args_list[-2].args[0] is storage.networks[0]
    assert add_network_tab.call_args_list[-1].args[0] is storage.networks[1]
