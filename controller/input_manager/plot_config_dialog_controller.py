from typing import List

from model.data.plot_generation_config import PlotGenerationConfig
from view.dialogs.plot_config_dialog import PlotConfigDialog

class PlotConfigDialogController:
    current_plot_config_dialog: PlotConfigDialog
    plots: List[PlotGenerationConfig]
    selected: int

    def __init__(self, current_plot_config_dialog: PlotConfigDialog):
        self.current_plot_config_dialog = current_plot_config_dialog
        self.plots = []
        self.selected = 0

    def select(self, index: int):
        pass

    def delete_selected(self):
        pass

    def edit_selected(self):
        pass

    def add_new_config(self):
        pass