from __future__ import annotations
from typing import TYPE_CHECKING

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.utils.result import Result
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType
from nn_verification_visualisation.view.dialogs.neuron_picker import NeuronPicker
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.view.dialogs.list_dialog_base import ListDialogBase, T

if TYPE_CHECKING:
    from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController

class PlotConfigDialog(ListDialogBase[PlotGenerationConfig]):
    parent_controller: PlotViewController

    def __init__(self, controller: PlotViewController):
        super().__init__(controller.current_plot_view.close_dialog, "Create Neuron Pairs", [], True)
        self.parent_controller = controller

    def on_confirm_clicked(self):
        self.on_close()

    def get_title(self, item: PlotGenerationConfig) -> str:
        return "Plot: " + item.algorithm.name

    def on_add_clicked(self):
        def on_neuron_picker_close():
            self.parent_controller.current_plot_view.close_dialog()
            config_res: Result[PlotGenerationConfig] = neuron_picker.construct_config()
            if not config_res.is_success:
                self.__show_error_dialog("Could not create configuration:", config_res.error)
                return
            self.add_item(config_res.data)

        neuron_picker = NeuronPicker(on_neuron_picker_close)

        self.parent_controller.current_plot_view.open_dialog(neuron_picker)

    def on_remove_clicked(self, item: PlotGenerationConfig, index: int) -> bool:
        return True

    def on_edit_clicked(self, item: PlotGenerationConfig) -> None:
        def on_neuron_picker_close():
            self.parent_controller.current_plot_view.close_dialog()
            edited_config_res: Result[PlotGenerationConfig] = neuron_picker.construct_config()
            if not edited_config_res.is_success:
                self.__show_error_dialog("Could not edit configuration:", edited_config_res.error)
                return
            index = self.list_widget.currentRow()
            if index >= 0:
                self.list_widget.takeItem(index)
                self.data.pop(index)
                self.list_widget.insertItem(index, self.get_title(edited_config_res.data))
                self.data.insert(index, edited_config_res.data)
                self.list_widget.setCurrentRow(index)

        neuron_picker = NeuronPicker(on_neuron_picker_close, preset=item)
        self.parent_controller.current_plot_view.open_dialog(neuron_picker)

        return None

    def __show_error_dialog(self, base_message: str, error: BaseException):
        has_message = error is not None and hasattr(error, 'message')
        error_message = error.message if has_message else str(error)
        error_dialog = InfoPopup(
            self.parent_controller.current_plot_view.close_dialog,
            base_message + "\n" + error_message,
            InfoType.ERROR
        )
        self.parent_controller.current_plot_view.open_dialog(error_dialog)