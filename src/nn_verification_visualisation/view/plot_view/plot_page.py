from typing import List

from PySide6.QtWidgets import QWidget

from model.data.diagram_config import DiagramConfig
from view.plot_view.plot_widget import PlotWidget
from view.base_view.tab import Tab


class PlotPage(Tab):
    configuration: DiagramConfig
    plots: List[PlotWidget]
    locked: List[PlotWidget]

    def __init__(self, configuration: DiagramConfig):
        super().__init__("Comparison")
        self.configuration = configuration

    def change_lock(self, widget: PlotWidget):
        pass

    def fullscreen(self, widget: PlotWidget):
        pass

    def export(self, widget: PlotWidget):
        pass

    def transform_changed(self, widget: PlotWidget, transform: tuple[float, float, float, float]):
        pass
