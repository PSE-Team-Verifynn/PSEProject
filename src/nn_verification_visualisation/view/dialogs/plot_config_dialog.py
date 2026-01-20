from typing import TYPE_CHECKING

from view.dialogs.neuron_picker import NeuronPicker

if TYPE_CHECKING:
    from controller.input_manager.plot_view_controller import PlotViewController

from model.data.plot_generation_config import PlotGenerationConfig
from view.dialogs.list_dialog_base import ListDialogBase, T

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
            config = neuron_picker.get_current_config()
            if config is None:
                return
            self.add_item(config)

        neuron_picker = NeuronPicker(on_neuron_picker_close)

        self.parent_controller.current_plot_view.open_dialog(neuron_picker)