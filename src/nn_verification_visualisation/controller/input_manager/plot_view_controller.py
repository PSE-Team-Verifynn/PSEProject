from __future__ import annotations

from typing import TYPE_CHECKING

from nn_verification_visualisation.model.data.algorithm_file_observer import AlgorithmFileObserver
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.view.dialogs.plot_config_dialog import PlotConfigDialog

if TYPE_CHECKING:
    from nn_verification_visualisation.view.plot_view.plot_view import PlotView

class PlotViewController:
    current_plot_view: PlotView
    current_tab: int

    def __init__(self, current_plot_view: PlotView):
        self.current_plot_view = current_plot_view
        self.current_tab = 0

        #start listening for algorithm changes
        AlgorithmFileObserver()

    def change_plot(self, plot_index: int, add: bool, pair_index: int):
        pass

    def start_computation(self, config: DiagramConfig):
        pass

    def change_tab(self, index: int):
        pass

    def export_plot(self, plot: Plot):
        pass

    def open_plot_generation_dialog(self):
        dialog = PlotConfigDialog(self)
        self.current_plot_view.open_dialog(dialog)