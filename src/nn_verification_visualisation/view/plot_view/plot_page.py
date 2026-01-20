from typing import List

from PySide6.QtWidgets import QWidget

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget
from nn_verification_visualisation.view.base_view.tab import Tab

class PlotPage(Tab):
    configuration: DiagramConfig
    plots: List[PlotWidget]
    locked: List[PlotWidget]

    def __init__(self, configuration: DiagramConfig):
        super().__init__("Example Tab")
        # configuration is currently not implemented
        # self.configuration = configuration

    def get_content(self) -> QWidget:
        # add content here
        return QWidget()

    def get_side_bar(self) -> QWidget:
        # add sidebar here
        return QWidget()

    def change_lock(self, widget: PlotWidget):
        pass

    def fullscreen(self, widget: PlotWidget):
        pass

    def export(self, widget: PlotWidget):
        pass

    def transform_changed(self, widget: PlotWidget, transform: tuple[float, float, float, float]):
        pass
