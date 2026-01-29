from __future__ import annotations
from typing import TYPE_CHECKING

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.dialogs.neuron_picker import NeuronPicker
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.view.dialogs.list_dialog_base import ListDialogBase

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
            config: PlotGenerationConfig = neuron_picker.construct_config()
            if config is None:
                return
            self.add_item(config)

        neuron_picker = NeuronPicker(on_neuron_picker_close)

        self.parent_controller.current_plot_view.open_dialog(neuron_picker)