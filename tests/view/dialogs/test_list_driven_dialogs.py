from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from PySide6.QtWidgets import QPushButton

from nn_verification_visualisation.utils.result import Success, Failure
from nn_verification_visualisation.view.dialogs.network_management_dialog import NetworkManagementDialog
from nn_verification_visualisation.view.dialogs.plot_config_dialog import PlotConfigDialog


def _make_network(name: str):
    return SimpleNamespace(network=SimpleNamespace(name=name))


def _make_plot_config(title: str):
    config = MagicMock()
    config.get_title.return_value = title
    return config


def _make_plot_controller():
    current_plot_view = MagicMock()
    current_plot_view.close_dialog = Mock()
    current_plot_view.open_dialog = Mock()
    controller = MagicMock()
    controller.current_plot_view = current_plot_view
    return controller


def test_network_management_dialog_uses_storage_names_and_adds_loaded_network(qapp):
    controller = MagicMock()
    controller.current_network_view.close_dialog = Mock()
    controller.load_new_network.return_value = Success(_make_network("Gamma"))

    with patch("nn_verification_visualisation.view.dialogs.network_management_dialog.Storage") as MockStorage:
        MockStorage.return_value.networks = [_make_network("Alpha"), _make_network("Beta")]

        dialog = NetworkManagementDialog(controller)
        dialog.on_add_clicked()

    assert dialog.list_widget.count() == 3
    assert dialog.list_widget.item(0).text() == "Alpha"
    assert dialog.list_widget.item(2).text() == "Gamma"


def test_network_management_dialog_remove_and_confirm_delegate_to_controller(qapp):
    controller = MagicMock()
    controller.current_network_view.close_dialog = Mock()

    with patch("nn_verification_visualisation.view.dialogs.network_management_dialog.Storage") as MockStorage:
        item = _make_network("Alpha")
        MockStorage.return_value.networks = [item]
        dialog = NetworkManagementDialog(controller)

    assert dialog.on_remove_clicked(item, 0) is controller.remove_neural_network.return_value
    dialog.on_confirm_clicked()

    controller.remove_neural_network.assert_called_once_with(item)
    controller.current_network_view.close_dialog.assert_called_once()


def test_plot_config_dialog_confirm_without_preset_starts_computation(qapp):
    controller = _make_plot_controller()
    config = _make_plot_config("Pair A")

    dialog = PlotConfigDialog(controller)
    dialog.data.append(config)
    dialog.on_confirm_clicked()

    controller.current_plot_view.close_dialog.assert_called_once()
    controller.start_computation.assert_called_once_with([config])


def test_plot_config_dialog_confirm_with_preset_opens_warning_dialog(qapp):
    controller = _make_plot_controller()
    on_accept = Mock()
    config = _make_plot_config("Pair A")

    with patch("nn_verification_visualisation.view.dialogs.plot_config_dialog.InfoPopup") as MockInfoPopup:
        popup = Mock()
        MockInfoPopup.return_value = popup

        dialog = PlotConfigDialog(controller, preset=([config], on_accept))
        dialog.on_confirm_clicked()

        controller.current_plot_view.open_dialog.assert_called_once_with(popup)
        buttons = MockInfoPopup.call_args.args[3]
        confirm_button = next(button for button in buttons if isinstance(button, QPushButton) and button.text() == "Continue")
        confirm_button.click()

    controller.start_computation.assert_called_once_with([config])
    on_accept.assert_called_once()


def test_plot_config_dialog_add_success_appends_constructed_config(qapp):
    controller = _make_plot_controller()
    produced = _make_plot_config("Created Pair")

    class FakePicker:
        def __init__(self, on_close, preset=None):
            self._on_close = on_close

        def construct_config(self):
            return Success(produced)

    with patch("nn_verification_visualisation.view.dialogs.plot_config_dialog.NeuronPicker", FakePicker):
        dialog = PlotConfigDialog(controller)
        dialog.on_add_clicked()
        picker = controller.current_plot_view.open_dialog.call_args.args[0]
        picker._on_close()

    assert dialog.data == [produced]
    assert dialog.list_widget.item(0).text() == "Created Pair"
    controller.current_plot_view.close_dialog.assert_called_once()


def test_plot_config_dialog_add_failure_opens_error_popup(qapp):
    controller = _make_plot_controller()

    class FakePicker:
        def __init__(self, on_close, preset=None):
            self._on_close = on_close

        def construct_config(self):
            return Failure(ValueError("bad config"))

    with (
        patch("nn_verification_visualisation.view.dialogs.plot_config_dialog.NeuronPicker", FakePicker),
        patch("nn_verification_visualisation.view.dialogs.plot_config_dialog.InfoPopup") as MockInfoPopup,
    ):
        dialog = PlotConfigDialog(controller)
        dialog.on_add_clicked()
        picker = controller.current_plot_view.open_dialog.call_args.args[0]
        picker._on_close()

    assert MockInfoPopup.called
    assert controller.current_plot_view.open_dialog.call_count == 2
