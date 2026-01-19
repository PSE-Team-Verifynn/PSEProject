from typing import TYPE_CHECKING

from view.dialogs.neuron_picker import NeuronPicker

if TYPE_CHECKING:
    from controller.input_manager.plot_view_controller import PlotViewController

from model.data.plot_generation_config import PlotGenerationConfig
from view.dialogs.list_dialog_base import ListDialogBase

class PlotConfigDialog(ListDialogBase[PlotGenerationConfig]):
    controller: PlotViewController

    def __init__(self, controller: PlotViewController):
        super().__init__(controller.current_plot_view.close_dialog, "Create Neuron Pairs", [], True)
        self.controller = controller

    def on_confirm_clicked(self):
        self.on_close()

    def on_add_clicked(self):

        def on_neuron_picker_close():
            self.controller.current_plot_view.close_dialog()
            # todo: get item from neuron_picker and add it to the list

        neuron_picker = NeuronPicker(on_neuron_picker_close)

        self.controller.current_plot_view.open_dialog(neuron_picker)